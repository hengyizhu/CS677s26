#include "vj/face_vision_engine.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cub/cub.cuh>

namespace vj {
namespace {

constexpr int kScanThreads = 256;
constexpr int kNormThreads = 256;

inline int divUpHost(int a, int b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __forceinline__ T ldgRead(const T* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(p);
#else
    return *p;
#endif
}

template <int BLOCK_THREADS>
__global__ void RowPrefixKernel(const uint8_t* __restrict__ d_gray,
                                int imageW,
                                int imageH,
                                int imageStride,
                                int nSamples,
                                int32_t* __restrict__ d_rowPrefix,
                                int32_t* __restrict__ d_rowPrefixSq,
                                int rowStride) {
    const int tid = static_cast<int>(threadIdx.x);
    const int rowIdx = static_cast<int>(blockIdx.x);  // [0, nSamples*imageH)
    if (rowIdx >= nSamples * imageH) {
        return;
    }

    const int sample = rowIdx / imageH;
    const int y = rowIdx - sample * imageH;

    const uint8_t* srcRow = d_gray + static_cast<size_t>(sample) * imageH * imageStride + y * imageStride;
    int32_t* dstRow = d_rowPrefix + (static_cast<size_t>(sample) * imageH + y) * rowStride;
    int32_t* dstRowSq = d_rowPrefixSq + (static_cast<size_t>(sample) * imageH + y) * rowStride;

    using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ int carry;
    __shared__ int carrySq;

    if (tid == 0) {
        dstRow[0] = 0;
        dstRowSq[0] = 0;
        carry = 0;
        carrySq = 0;
    }
    __syncthreads();

    // 每个 tile 处理一段连续像素，先做行方向扫描，再叠加 tile 的 carry。
    for (int base = 0; base < imageW; base += BLOCK_THREADS) {
        const int x = base + tid;
        int pix = 0;
        if (x < imageW) {
            pix = static_cast<int>(srcRow[x]);
        }

        int scanVal = 0;
        int blockAgg = 0;
        BlockScan(scanStorage).InclusiveSum(pix, scanVal, blockAgg);
        const int pref = scanVal + carry;
        __syncthreads();
        if (x < imageW) {
            dstRow[x + 1] = pref;
        }
        if (tid == 0) {
            carry += blockAgg;
        }
        __syncthreads();

        const int pixSq = pix * pix;
        int scanSq = 0;
        int blockAggSq = 0;
        BlockScan(scanStorage).InclusiveSum(pixSq, scanSq, blockAggSq);
        const int prefSq = scanSq + carrySq;
        __syncthreads();
        if (x < imageW) {
            dstRowSq[x + 1] = prefSq;
        }
        if (tid == 0) {
            carrySq += blockAggSq;
        }
        __syncthreads();
    }
}

__global__ void ColumnScanTransposeKernel(const int32_t* __restrict__ d_rowPrefix,
                                          const int32_t* __restrict__ d_rowPrefixSq,
                                          int imageW,
                                          int imageH,
                                          int nSamples,
                                          int rowStride,
                                          int32_t* __restrict__ d_sumT,
                                          int32_t* __restrict__ d_sqsumT) {
    // 线程 x 维度走 sample，保证写 dOut[pixel*N + sample] 时 sample 连续，合并写。
    const int sample = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int x = static_cast<int>(blockIdx.y);

    if (sample >= nSamples || x > imageW) {
        return;
    }

    int64_t colSum = 0;
    int64_t colSqSum = 0;

    // y=0 padding row
    {
        const size_t p0 = static_cast<size_t>(x);
        d_sumT[p0 * nSamples + sample] = 0;
        d_sqsumT[p0 * nSamples + sample] = 0;
    }

    // 列扫描 + 转置写回: dOut[pixel_idx * N + sample_idx]
    for (int y = 1; y <= imageH; ++y) {
        const size_t rowIdx = (static_cast<size_t>(sample) * imageH + (y - 1)) * rowStride + x;
        colSum += d_rowPrefix[rowIdx];
        colSqSum += d_rowPrefixSq[rowIdx];

        const size_t p = static_cast<size_t>(y) * (imageW + 1) + x;
        d_sumT[p * nSamples + sample] = static_cast<int32_t>(colSum);
        d_sqsumT[p * nSamples + sample] = static_cast<int32_t>(colSqSum);
    }
}

__global__ void ColumnScanTransposeSingleSampleKernel(const int32_t* __restrict__ d_rowPrefix,
                                                      const int32_t* __restrict__ d_rowPrefixSq,
                                                      int imageW,
                                                      int imageH,
                                                      int rowStride,
                                                      int32_t* __restrict__ d_sumT,
                                                      int32_t* __restrict__ d_sqsumT) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (x > imageW) {
        return;
    }

    int64_t colSum = 0;
    int64_t colSqSum = 0;

    d_sumT[x] = 0;
    d_sqsumT[x] = 0;

    for (int y = 1; y <= imageH; ++y) {
        const size_t rowIdx = static_cast<size_t>(y - 1) * rowStride + x;
        colSum += d_rowPrefix[rowIdx];
        colSqSum += d_rowPrefixSq[rowIdx];
        const size_t p = static_cast<size_t>(y) * (imageW + 1) + x;
        d_sumT[p] = static_cast<int32_t>(colSum);
        d_sqsumT[p] = static_cast<int32_t>(colSqSum);
    }
}

__global__ void ComputeInvNormKernel(const int32_t* __restrict__ d_sumT,
                                     const int32_t* __restrict__ d_sqsumT,
                                     int nSamples,
                                     int iiWidth,
                                     int iiHeight,
                                     float* __restrict__ d_invNorm) {
    const int sample = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (sample >= nSamples) {
        return;
    }

    if (iiWidth < 3 || iiHeight < 3) {
        d_invNorm[sample] = 1.0f;
        return;
    }

    const int x = 1;
    const int y = 1;
    const int w = iiWidth - 3;
    const int h = iiHeight - 3;
    const int area = w * h;

    const int p0 = x + iiWidth * y;
    const int p1 = (x + w) + iiWidth * y;
    const int p2 = x + iiWidth * (y + h);
    const int p3 = (x + w) + iiWidth * (y + h);

    const auto readSum = [&](int p) -> double {
        return static_cast<double>(d_sumT[static_cast<size_t>(p) * nSamples + sample]);
    };
    const auto readSq = [&](int p) -> double {
        return static_cast<double>(d_sqsumT[static_cast<size_t>(p) * nSamples + sample]);
    };

    const double s = readSum(p0) - readSum(p1) - readSum(p2) + readSum(p3);
    const double sq = readSq(p0) - readSq(p1) - readSq(p2) + readSq(p3);

    const double nf2 = static_cast<double>(area) * sq - s * s;
    d_invNorm[sample] = (nf2 > 1e-12) ? static_cast<float>(1.0 / std::sqrt(nf2)) : 0.0f;
}

__global__ void ResizeBilinearKernel(const uint8_t* __restrict__ d_src,
                                     int srcW,
                                     int srcH,
                                     int srcStride,
                                     uint8_t* __restrict__ d_dst,
                                     int dstW,
                                     int dstH,
                                     int dstStride,
                                     float scale) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dstW || y >= dstH) {
        return;
    }

    // Match OpenCV resize geometry: map destination pixel centers to source space.
    const float fx = (static_cast<float>(x) + 0.5f) * scale - 0.5f;
    const float fy = (static_cast<float>(y) + 0.5f) * scale - 0.5f;

    const int sx0 = max(0, min(static_cast<int>(floorf(fx)), srcW - 1));
    const int sy0 = max(0, min(static_cast<int>(floorf(fy)), srcH - 1));
    const int sx1 = min(sx0 + 1, srcW - 1);
    const int sy1 = min(sy0 + 1, srcH - 1);

    const float ax = fx - static_cast<float>(sx0);
    const float ay = fy - static_cast<float>(sy0);

    const float p00 = static_cast<float>(d_src[sy0 * srcStride + sx0]);
    const float p01 = static_cast<float>(d_src[sy0 * srcStride + sx1]);
    const float p10 = static_cast<float>(d_src[sy1 * srcStride + sx0]);
    const float p11 = static_cast<float>(d_src[sy1 * srcStride + sx1]);

    const float top = p00 + (p01 - p00) * ax;
    const float bot = p10 + (p11 - p10) * ax;
    const float val = top + (bot - top) * ay;
    d_dst[y * dstStride + x] = static_cast<uint8_t>(fminf(fmaxf(val + 0.5f, 0.0f), 255.0f));
}

__global__ void BuildFeatureLUTKernel(const HaarFeature* __restrict__ d_features,
                                      int featureCount,
                                      int modelIIWidth,
                                      int32_t* __restrict__ dR0Dx0,
                                      int32_t* __restrict__ dR0Dy0,
                                      int32_t* __restrict__ dR0Dx1,
                                      int32_t* __restrict__ dR0Dy1,
                                      int32_t* __restrict__ dR0Dx2,
                                      int32_t* __restrict__ dR0Dy2,
                                      int32_t* __restrict__ dR0Dx3,
                                      int32_t* __restrict__ dR0Dy3,
                                      float* __restrict__ dR0W,
                                      int32_t* __restrict__ dR1Dx0,
                                      int32_t* __restrict__ dR1Dy0,
                                      int32_t* __restrict__ dR1Dx1,
                                      int32_t* __restrict__ dR1Dy1,
                                      int32_t* __restrict__ dR1Dx2,
                                      int32_t* __restrict__ dR1Dy2,
                                      int32_t* __restrict__ dR1Dx3,
                                      int32_t* __restrict__ dR1Dy3,
                                      float* __restrict__ dR1W,
                                      int32_t* __restrict__ dR2Dx0,
                                      int32_t* __restrict__ dR2Dy0,
                                      int32_t* __restrict__ dR2Dx1,
                                      int32_t* __restrict__ dR2Dy1,
                                      int32_t* __restrict__ dR2Dx2,
                                      int32_t* __restrict__ dR2Dy2,
                                      int32_t* __restrict__ dR2Dx3,
                                      int32_t* __restrict__ dR2Dy3,
                                      float* __restrict__ dR2W,
                                      uint8_t* __restrict__ dRectCount) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= featureCount) {
        return;
    }

    const HaarFeature f = d_features[idx];
    auto decode = [&](int p, int32_t& dx, int32_t& dy) {
        const int py = p / modelIIWidth;
        const int px = p - py * modelIIWidth;
        dx = px;
        dy = py;
    };

    decode(f.r0.p0, dR0Dx0[idx], dR0Dy0[idx]);
    decode(f.r0.p1, dR0Dx1[idx], dR0Dy1[idx]);
    decode(f.r0.p2, dR0Dx2[idx], dR0Dy2[idx]);
    decode(f.r0.p3, dR0Dx3[idx], dR0Dy3[idx]);
    dR0W[idx] = f.r0.w;

    decode(f.r1.p0, dR1Dx0[idx], dR1Dy0[idx]);
    decode(f.r1.p1, dR1Dx1[idx], dR1Dy1[idx]);
    decode(f.r1.p2, dR1Dx2[idx], dR1Dy2[idx]);
    decode(f.r1.p3, dR1Dx3[idx], dR1Dy3[idx]);
    dR1W[idx] = f.r1.w;

    decode(f.r2.p0, dR2Dx0[idx], dR2Dy0[idx]);
    decode(f.r2.p1, dR2Dx1[idx], dR2Dy1[idx]);
    decode(f.r2.p2, dR2Dx2[idx], dR2Dy2[idx]);
    decode(f.r2.p3, dR2Dx3[idx], dR2Dy3[idx]);
    dR2W[idx] = f.r2.w;

    dRectCount[idx] = f.rectCount;
}

__global__ __launch_bounds__(256, 2) void DetectCascadeKernel(const int32_t* __restrict__ d_sumT,
                                                               const int32_t* __restrict__ d_sqsumT,
                                                               int iiWidth,
                                                               int iiHeight,
                                                               const int32_t* __restrict__ dR0Dx0,
                                                               const int32_t* __restrict__ dR0Dy0,
                                                               const int32_t* __restrict__ dR0Dx1,
                                                               const int32_t* __restrict__ dR0Dy1,
                                                               const int32_t* __restrict__ dR0Dx2,
                                                               const int32_t* __restrict__ dR0Dy2,
                                                               const int32_t* __restrict__ dR0Dx3,
                                                               const int32_t* __restrict__ dR0Dy3,
                                                               const float* __restrict__ dR0W,
                                                               const int32_t* __restrict__ dR1Dx0,
                                                               const int32_t* __restrict__ dR1Dy0,
                                                               const int32_t* __restrict__ dR1Dx1,
                                                               const int32_t* __restrict__ dR1Dy1,
                                                               const int32_t* __restrict__ dR1Dx2,
                                                               const int32_t* __restrict__ dR1Dy2,
                                                               const int32_t* __restrict__ dR1Dx3,
                                                               const int32_t* __restrict__ dR1Dy3,
                                                               const float* __restrict__ dR1W,
                                                               const int32_t* __restrict__ dR2Dx0,
                                                               const int32_t* __restrict__ dR2Dy0,
                                                               const int32_t* __restrict__ dR2Dx1,
                                                               const int32_t* __restrict__ dR2Dy1,
                                                               const int32_t* __restrict__ dR2Dx2,
                                                               const int32_t* __restrict__ dR2Dy2,
                                                               const int32_t* __restrict__ dR2Dx3,
                                                               const int32_t* __restrict__ dR2Dy3,
                                                               const float* __restrict__ dR2W,
                                                               const uint8_t* __restrict__ dRectCount,
                                                               const GpuStump4* __restrict__ d_stumps,
                                                               const CascadeStage* __restrict__ d_stages,
                                                               int stageCount,
                                                               int winW,
                                                               int winH,
                                                               int scaleIdx,
                                                               float scale,
                                                               int step,
                                                               Detection* __restrict__ d_out,
                                                               int* __restrict__ d_count,
                                                               int maxOut) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * step;
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) * step;
    const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);

    const int workW = iiWidth - 1 - winW + 1;
    const int workH = iiHeight - 1 - winH + 1;
    const bool inRange = (x < workW && y < workH);
    const int base = y * iiWidth + x;

    const int nx = 1;
    const int ny = 1;
    const int nw = winW - 2;
    const int nh = winH - 2;
    const int narea = nw * nh;

    const int np0 = nx + iiWidth * ny;
    const int np1 = (nx + nw) + iiWidth * ny;
    const int np2 = nx + iiWidth * (ny + nh);
    const int np3 = (nx + nw) + iiWidth * (ny + nh);

    const auto readII = [&](int p) -> float { return static_cast<float>(ldgRead(d_sumT + p)); };
    const auto readSqU32 = [&](int p) -> uint32_t {
        // Match OpenCV setWindow(): sqsum delta is interpreted as unsigned 32-bit.
        return static_cast<uint32_t>(ldgRead(d_sqsumT + p));
    };

    bool pass = false;
    float lastStageScore = 0.0f;
    if (inRange) {
        const double normSum = static_cast<double>(readII(base + np0) - readII(base + np1) - readII(base + np2) + readII(base + np3));
        const uint32_t normSqU =
            readSqU32(base + np0) - readSqU32(base + np1) - readSqU32(base + np2) + readSqU32(base + np3);
        const double normSq = static_cast<double>(normSqU);
        const double nf2 = static_cast<double>(narea) * normSq - normSum * normSum;
        if (nf2 > 0.0) {
            const float invNorm = static_cast<float>(1.0 / sqrt(nf2));
            // Match OpenCV HaarEvaluator::setWindow gate for low-texture windows.
            if (static_cast<double>(narea) * static_cast<double>(invNorm) < 1e-1) {
                pass = true;
                // Cascade early rejection: once one stage fails, this window is rejected.
                for (int si = 0; si < stageCount; ++si) {
                    const CascadeStage st = d_stages[si];
                    float stageSum = 0.0f;

                    for (int wi = 0; wi < st.ntrees; ++wi) {
                        const float4 s = ldgRead(&(d_stumps[st.first + wi].st));
                        const int featureIdx = __float_as_int(s.x);
                        const float theta = s.y;
                        const float leftVal = s.z;
                        const float rightVal = s.w;

                        const int r0p0 = ldgRead(dR0Dy0 + featureIdx) * iiWidth + ldgRead(dR0Dx0 + featureIdx);
                        const int r0p1 = ldgRead(dR0Dy1 + featureIdx) * iiWidth + ldgRead(dR0Dx1 + featureIdx);
                        const int r0p2 = ldgRead(dR0Dy2 + featureIdx) * iiWidth + ldgRead(dR0Dx2 + featureIdx);
                        const int r0p3 = ldgRead(dR0Dy3 + featureIdx) * iiWidth + ldgRead(dR0Dx3 + featureIdx);
                        const int r1p0 = ldgRead(dR1Dy0 + featureIdx) * iiWidth + ldgRead(dR1Dx0 + featureIdx);
                        const int r1p1 = ldgRead(dR1Dy1 + featureIdx) * iiWidth + ldgRead(dR1Dx1 + featureIdx);
                        const int r1p2 = ldgRead(dR1Dy2 + featureIdx) * iiWidth + ldgRead(dR1Dx2 + featureIdx);
                        const int r1p3 = ldgRead(dR1Dy3 + featureIdx) * iiWidth + ldgRead(dR1Dx3 + featureIdx);

                        const float r0w = ldgRead(dR0W + featureIdx);
                        const float r1w = ldgRead(dR1W + featureIdx);

                        const float v0 = (readII(base + r0p0) - readII(base + r0p1) - readII(base + r0p2) + readII(base + r0p3)) * r0w;
                        const float v1 = (readII(base + r1p0) - readII(base + r1p1) - readII(base + r1p2) + readII(base + r1p3)) * r1w;

                        float raw = v0 + v1;
                        if (ldgRead(dRectCount + featureIdx) == 3) {
                            const int r2p0 = ldgRead(dR2Dy0 + featureIdx) * iiWidth + ldgRead(dR2Dx0 + featureIdx);
                            const int r2p1 = ldgRead(dR2Dy1 + featureIdx) * iiWidth + ldgRead(dR2Dx1 + featureIdx);
                            const int r2p2 = ldgRead(dR2Dy2 + featureIdx) * iiWidth + ldgRead(dR2Dx2 + featureIdx);
                            const int r2p3 = ldgRead(dR2Dy3 + featureIdx) * iiWidth + ldgRead(dR2Dx3 + featureIdx);
                            const float r2w = ldgRead(dR2W + featureIdx);
                            const float v2 = (readII(base + r2p0) - readII(base + r2p1) - readII(base + r2p2) + readII(base + r2p3)) * r2w;
                            raw += v2;
                        }

                        // Keep decision numerically identical to training: compare normalized response.
                        stageSum += (raw * invNorm < theta) ? leftVal : rightVal;
                    }

                    if (stageSum < st.threshold) {
                        pass = false;
                        break;
                    }
                    lastStageScore = stageSum;
                }
            }
        }
    }

    __shared__ int sCount;
    __shared__ int sBase;
    if (tid == 0) {
        sCount = 0;
        sBase = 0;
    }
    __syncthreads();

    int localRank = -1;
    if (pass) {
        localRank = atomicAdd(&sCount, 1);
    }
    __syncthreads();

    if (tid == 0 && sCount > 0) {
        sBase = atomicAdd(d_count, sCount);
    }
    __syncthreads();

    if (pass) {
        const int outIdx = sBase + localRank;
        if (outIdx < maxOut) {
            Detection det{};
            det.x = static_cast<int>(x * scale + 0.5f);
            det.y = static_cast<int>(y * scale + 0.5f);
            det.w = static_cast<int>(winW * scale + 0.5f);
            det.h = static_cast<int>(winH * scale + 0.5f);
            det.scaleIdx = scaleIdx;
            det.score = lastStageScore;
            d_out[outIdx] = det;
        }
    }
}

inline float iou(const Detection& a, const Detection& b) {
    const int x1 = max(a.x, b.x);
    const int y1 = max(a.y, b.y);
    const int x2 = min(a.x + a.w, b.x + b.w);
    const int y2 = min(a.y + a.h, b.y + b.h);
    const int iw = max(0, x2 - x1);
    const int ih = max(0, y2 - y1);
    const float inter = static_cast<float>(iw * ih);
    const float uni = static_cast<float>(a.w * a.h + b.w * b.h) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

struct DisjointSet {
    std::vector<int> p;
    std::vector<int> r;

    explicit DisjointSet(int n) : p(n), r(n, 0) {
        for (int i = 0; i < n; ++i) p[i] = i;
    }
    int find(int x) {
        if (p[x] != x) p[x] = find(p[x]);
        return p[x];
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (r[a] < r[b]) std::swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) ++r[a];
    }
};

inline bool similarRects(const Detection& a, const Detection& b, float eps) {
    const int delta = static_cast<int>(
        eps * (std::min(a.w, b.w) + std::min(a.h, b.h)) * 0.5f);
    return std::abs(a.x - b.x) <= delta &&
           std::abs(a.y - b.y) <= delta &&
           std::abs((a.x + a.w) - (b.x + b.w)) <= delta &&
           std::abs((a.y + a.h) - (b.y + b.h)) <= delta;
}

inline FastRect makeFastRect(int x, int y, int w, int h, int iiWidth, float weight, bool tilted) {
    FastRect fr{};
    if (!tilted) {
        fr.p0 = x + iiWidth * y;
        fr.p1 = (x + w) + iiWidth * y;
        fr.p2 = x + iiWidth * (y + h);
        fr.p3 = (x + w) + iiWidth * (y + h);
    } else {
        // Same as CV_TILTED_OFFSETS
        fr.p0 = x + iiWidth * y;
        fr.p1 = (x - h) + iiWidth * (y + h);
        fr.p2 = (x + w) + iiWidth * (y + w);
        fr.p3 = (x + w - h) + iiWidth * (y + w + h);
    }
    fr.w = weight;
    return fr;
}

} // namespace

Status FaceVisionEngine::fromCuda(cudaError_t err) {
    return err == cudaSuccess ? Status::kOk : Status::kCudaError;
}

FaceVisionEngine::FaceVisionEngine(const WindowSpec& win,
                                   int maxSamples,
                                   int maxImageW,
                                   int maxImageH,
                                   bool enableTilted,
                                   int maxDetections)
    : win_(win),
      maxSamples_(maxSamples),
      maxImageW_(maxImageW),
      maxImageH_(maxImageH),
      enableTilted_(enableTilted),
      maxDetections_(maxDetections) {
    if (maxSamples_ <= 0 || maxImageW_ <= 0 || maxImageH_ <= 0 || maxDetections_ <= 0) {
        throw std::invalid_argument("FaceVisionEngine: invalid constructor args");
    }

    const int rowStrideMax = maxImageW_ + 1;
    const size_t rowElems = static_cast<size_t>(maxSamples_) * maxImageH_ * rowStrideMax;
    const size_t iiElems = static_cast<size_t>(maxSamples_) * (maxImageW_ + 1) * (maxImageH_ + 1);

    auto alloc_or_throw = [](void** ptr, size_t bytes, const char* msg) {
        const cudaError_t st = cudaMalloc(ptr, bytes);
        if (st != cudaSuccess) {
            throw std::runtime_error(msg);
        }
    };

    try {
        alloc_or_throw(reinterpret_cast<void**>(&dRowPrefix_), rowElems * sizeof(int32_t),
                       "FaceVisionEngine: cudaMalloc dRowPrefix_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dRowPrefixSq_), rowElems * sizeof(int32_t),
                       "FaceVisionEngine: cudaMalloc dRowPrefixSq_ failed");

        alloc_or_throw(reinterpret_cast<void**>(&dSumT_), iiElems * sizeof(int32_t),
                       "FaceVisionEngine: cudaMalloc dSumT_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dSqsumT_), iiElems * sizeof(int32_t),
                       "FaceVisionEngine: cudaMalloc dSqsumT_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dInvNorm_), static_cast<size_t>(maxSamples_) * sizeof(float),
                       "FaceVisionEngine: cudaMalloc dInvNorm_ failed");

        alloc_or_throw(reinterpret_cast<void**>(&dPyrBufA_), static_cast<size_t>(maxImageW_) * maxImageH_ * sizeof(uint8_t),
                       "FaceVisionEngine: cudaMalloc dPyrBufA_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dPyrBufB_), static_cast<size_t>(maxImageW_) * maxImageH_ * sizeof(uint8_t),
                       "FaceVisionEngine: cudaMalloc dPyrBufB_ failed");

        alloc_or_throw(reinterpret_cast<void**>(&dDetectOut_), static_cast<size_t>(maxDetections_) * sizeof(Detection),
                       "FaceVisionEngine: cudaMalloc dDetectOut_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dDetectCount_), sizeof(int),
                       "FaceVisionEngine: cudaMalloc dDetectCount_ failed");

        if (cudaMallocHost(reinterpret_cast<void**>(&hDetectPinned_),
                           static_cast<size_t>(maxDetections_) * sizeof(Detection)) != cudaSuccess) {
            throw std::runtime_error("FaceVisionEngine: cudaMallocHost hDetectPinned_ failed");
        }
        if (cudaMallocHost(reinterpret_cast<void**>(&hDetectCountPinned_), sizeof(int)) != cudaSuccess) {
            throw std::runtime_error("FaceVisionEngine: cudaMallocHost hDetectCountPinned_ failed");
        }
    } catch (...) {
        if (hDetectCountPinned_) cudaFreeHost(hDetectCountPinned_);
        if (hDetectPinned_) cudaFreeHost(hDetectPinned_);
        if (dDetectCount_) cudaFree(dDetectCount_);
        if (dDetectOut_) cudaFree(dDetectOut_);
        if (dPyrBufB_) cudaFree(dPyrBufB_);
        if (dPyrBufA_) cudaFree(dPyrBufA_);
        if (dInvNorm_) cudaFree(dInvNorm_);
        if (dSqsumT_) cudaFree(dSqsumT_);
        if (dSumT_) cudaFree(dSumT_);
        if (dRowPrefixSq_) cudaFree(dRowPrefixSq_);
        if (dRowPrefix_) cudaFree(dRowPrefix_);
        throw;
    }
}

FaceVisionEngine::~FaceVisionEngine() {
    for (int r = 0; r < kLutRects; ++r) {
        for (int c = 0; c < kLutCorners; ++c) {
            if (dLutDx_[r][c]) cudaFree(dLutDx_[r][c]);
            if (dLutDy_[r][c]) cudaFree(dLutDy_[r][c]);
        }
        if (dLutW_[r]) cudaFree(dLutW_[r]);
    }
    if (dLutRectCount_) cudaFree(dLutRectCount_);
    if (hDetectCountPinned_) cudaFreeHost(hDetectCountPinned_);
    if (hDetectPinned_) cudaFreeHost(hDetectPinned_);
    if (dDetectCount_) cudaFree(dDetectCount_);
    if (dDetectOut_) cudaFree(dDetectOut_);
    if (dModelStages_) cudaFree(dModelStages_);
    if (dModelStumps_) cudaFree(dModelStumps_);
    if (dModelFeatures_) cudaFree(dModelFeatures_);
    if (dPyrBufB_) cudaFree(dPyrBufB_);
    if (dPyrBufA_) cudaFree(dPyrBufA_);
    if (dInvNorm_) cudaFree(dInvNorm_);
    if (dSqsumT_) cudaFree(dSqsumT_);
    if (dSumT_) cudaFree(dSumT_);
    if (dRowPrefixSq_) cudaFree(dRowPrefixSq_);
    if (dRowPrefix_) cudaFree(dRowPrefix_);
    if (dFeaturePool_) cudaFree(dFeaturePool_);
}

FaceVisionEngine::FaceVisionEngine(FaceVisionEngine&& other) noexcept {
    *this = std::move(other);
}

FaceVisionEngine& FaceVisionEngine::operator=(FaceVisionEngine&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    for (int r = 0; r < kLutRects; ++r) {
        for (int c = 0; c < kLutCorners; ++c) {
            if (dLutDx_[r][c]) cudaFree(dLutDx_[r][c]);
            if (dLutDy_[r][c]) cudaFree(dLutDy_[r][c]);
        }
        if (dLutW_[r]) cudaFree(dLutW_[r]);
    }
    if (dLutRectCount_) cudaFree(dLutRectCount_);
    if (hDetectCountPinned_) cudaFreeHost(hDetectCountPinned_);
    if (hDetectPinned_) cudaFreeHost(hDetectPinned_);
    if (dDetectCount_) cudaFree(dDetectCount_);
    if (dDetectOut_) cudaFree(dDetectOut_);
    if (dModelStages_) cudaFree(dModelStages_);
    if (dModelStumps_) cudaFree(dModelStumps_);
    if (dModelFeatures_) cudaFree(dModelFeatures_);
    if (dPyrBufB_) cudaFree(dPyrBufB_);
    if (dPyrBufA_) cudaFree(dPyrBufA_);
    if (dInvNorm_) cudaFree(dInvNorm_);
    if (dSqsumT_) cudaFree(dSqsumT_);
    if (dSumT_) cudaFree(dSumT_);
    if (dRowPrefixSq_) cudaFree(dRowPrefixSq_);
    if (dRowPrefix_) cudaFree(dRowPrefix_);
    if (dFeaturePool_) cudaFree(dFeaturePool_);

    win_ = other.win_;
    maxSamples_ = other.maxSamples_;
    maxImageW_ = other.maxImageW_;
    maxImageH_ = other.maxImageH_;
    enableTilted_ = other.enableTilted_;
    maxDetections_ = other.maxDetections_;

    featurePoolSize_ = other.featurePoolSize_;
    dFeaturePool_ = other.dFeaturePool_;

    currentSamples_ = other.currentSamples_;
    currentImageW_ = other.currentImageW_;
    currentImageH_ = other.currentImageH_;

    dRowPrefix_ = other.dRowPrefix_;
    dRowPrefixSq_ = other.dRowPrefixSq_;
    dSumT_ = other.dSumT_;
    dSqsumT_ = other.dSqsumT_;
    dInvNorm_ = other.dInvNorm_;

    dPyrBufA_ = other.dPyrBufA_;
    dPyrBufB_ = other.dPyrBufB_;

    dModelFeatures_ = other.dModelFeatures_;
    for (int r = 0; r < kLutRects; ++r) {
        for (int c = 0; c < kLutCorners; ++c) {
            dLutDx_[r][c] = other.dLutDx_[r][c];
            dLutDy_[r][c] = other.dLutDy_[r][c];
        }
        dLutW_[r] = other.dLutW_[r];
    }
    dLutRectCount_ = other.dLutRectCount_;
    modelFeatureCount_ = other.modelFeatureCount_;
    modelFeatureCapacity_ = other.modelFeatureCapacity_;
    modelLutCapacity_ = other.modelLutCapacity_;
    dModelStumps_ = other.dModelStumps_;
    modelStumpCount_ = other.modelStumpCount_;
    modelStumpCapacity_ = other.modelStumpCapacity_;
    dModelStages_ = other.dModelStages_;
    modelStageCount_ = other.modelStageCount_;
    modelStageCapacity_ = other.modelStageCapacity_;

    dDetectOut_ = other.dDetectOut_;
    dDetectCount_ = other.dDetectCount_;
    hDetectPinned_ = other.hDetectPinned_;
    hDetectCountPinned_ = other.hDetectCountPinned_;

    other.dFeaturePool_ = nullptr;
    other.dRowPrefix_ = nullptr;
    other.dRowPrefixSq_ = nullptr;
    other.dSumT_ = nullptr;
    other.dSqsumT_ = nullptr;
    other.dInvNorm_ = nullptr;
    other.dPyrBufA_ = nullptr;
    other.dPyrBufB_ = nullptr;
    other.dModelFeatures_ = nullptr;
    for (int r = 0; r < kLutRects; ++r) {
        for (int c = 0; c < kLutCorners; ++c) {
            other.dLutDx_[r][c] = nullptr;
            other.dLutDy_[r][c] = nullptr;
        }
        other.dLutW_[r] = nullptr;
    }
    other.dLutRectCount_ = nullptr;
    other.dModelStumps_ = nullptr;
    other.dModelStages_ = nullptr;
    other.dDetectOut_ = nullptr;
    other.dDetectCount_ = nullptr;
    other.hDetectPinned_ = nullptr;
    other.hDetectCountPinned_ = nullptr;
    other.featurePoolSize_ = 0;
    other.currentSamples_ = 0;
    other.currentImageW_ = 0;
    other.currentImageH_ = 0;
    other.modelFeatureCount_ = 0;
    other.modelFeatureCapacity_ = 0;
    other.modelLutCapacity_ = 0;
    other.modelStumpCount_ = 0;
    other.modelStumpCapacity_ = 0;
    other.modelStageCount_ = 0;
    other.modelStageCapacity_ = 0;
    return *this;
}

int FaceVisionEngine::countHaarFeatures(const WindowSpec& win, FeatureMode mode) {
    return static_cast<int>(buildHaarFeaturePool(win, mode).size());
}

std::vector<HaarFeature> FaceVisionEngine::buildHaarFeaturePool(const WindowSpec& win, FeatureMode mode) {
    std::vector<HaarFeature> out;
    const int W = win.winW;
    const int H = win.winH;
    const int iiW = W + 1;
    const bool corePlus = mode != FeatureMode::Basic;
    const bool withTilted = mode == FeatureMode::All;

    out.reserve(180000);

    auto push2 = [&](bool tilted,
                     int x0, int y0, int w0, int h0, float wt0,
                     int x1, int y1, int w1, int h1, float wt1) {
        HaarFeature f{};
        f.r0 = makeFastRect(x0, y0, w0, h0, iiW, wt0, tilted);
        f.r1 = makeFastRect(x1, y1, w1, h1, iiW, wt1, tilted);
        f.r2 = FastRect{0, 0, 0, 0, 0.0f};
        f.rectCount = 2;
        f.tilted = tilted ? 1 : 0;
        f.reserved = 0;
        out.push_back(f);
    };

    auto push3 = [&](bool tilted,
                     int x0, int y0, int w0, int h0, float wt0,
                     int x1, int y1, int w1, int h1, float wt1,
                     int x2, int y2, int w2, int h2, float wt2) {
        HaarFeature f{};
        f.r0 = makeFastRect(x0, y0, w0, h0, iiW, wt0, tilted);
        f.r1 = makeFastRect(x1, y1, w1, h1, iiW, wt1, tilted);
        f.r2 = makeFastRect(x2, y2, w2, h2, iiW, wt2, tilted);
        f.rectCount = 3;
        f.tilted = tilted ? 1 : 0;
        f.reserved = 0;
        out.push_back(f);
    };

    // 与 OpenCV traincascade 逻辑一致：遍历位置、尺寸、模板类型。
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) {
            for (int dx = 1; dx <= W; ++dx) {
                for (int dy = 1; dy <= H; ++dy) {
                    // upright x2
                    if (x + 2 * dx <= W && y + dy <= H) {
                        push2(false, x, y, 2 * dx, dy, -1.0f,
                                   x + dx, y, dx, dy, +2.0f);
                    }
                    // upright y2
                    if (x + dx <= W && y + 2 * dy <= H) {
                        push2(false, x, y, dx, 2 * dy, -1.0f,
                                   x, y + dy, dx, dy, +2.0f);
                    }
                    // upright x3
                    if (x + 3 * dx <= W && y + dy <= H) {
                        push2(false, x, y, 3 * dx, dy, -1.0f,
                                   x + dx, y, dx, dy, +2.0f);
                    }
                    // upright y3
                    if (x + dx <= W && y + 3 * dy <= H) {
                        push2(false, x, y, dx, 3 * dy, -1.0f,
                                   x, y + dy, dx, dy, +2.0f);
                    }
                    // checker 2x2
                    if (x + 2 * dx <= W && y + 2 * dy <= H) {
                        push3(false, x, y, 2 * dx, 2 * dy, -1.0f,
                                   x, y, dx, dy, +2.0f,
                                   x + dx, y + dy, dx, dy, +2.0f);
                    }

                    if (corePlus) {
                        // upright x4
                        if (x + 4 * dx <= W && y + dy <= H) {
                            push2(false, x, y, 4 * dx, dy, -1.0f,
                                       x + dx, y, 2 * dx, dy, +2.0f);
                        }
                        // upright y4
                        if (x + dx <= W && y + 4 * dy <= H) {
                            push2(false, x, y, dx, 4 * dy, -1.0f,
                                       x, y + dy, dx, 2 * dy, +2.0f);
                        }
                        // center-surround 3x3
                        if (x + 3 * dx <= W && y + 3 * dy <= H) {
                            push2(false, x, y, 3 * dx, 3 * dy, -1.0f,
                                       x + dx, y + dy, dx, dy, +9.0f);
                        }
                    }

                    if (withTilted) {
                        // tilted x2
                        if (x + 2 * dx <= W && y + 2 * dx + dy <= H && x - dy >= 0) {
                            push2(true, x, y, 2 * dx, dy, -1.0f,
                                      x, y, dx, dy, +2.0f);
                        }
                        // tilted y2
                        if (x + dx <= W && y + dx + 2 * dy <= H && x - 2 * dy >= 0) {
                            push2(true, x, y, dx, 2 * dy, -1.0f,
                                      x, y, dx, dy, +2.0f);
                        }
                        // tilted x3
                        if (x + 3 * dx <= W && y + 3 * dx + dy <= H && x - dy >= 0) {
                            push2(true, x, y, 3 * dx, dy, -1.0f,
                                      x + dx, y + dx, dx, dy, +3.0f);
                        }
                        // tilted y3
                        if (x + dx <= W && y + dx + 3 * dy <= H && x - 3 * dy >= 0) {
                            push2(true, x, y, dx, 3 * dy, -1.0f,
                                      x - dy, y + dy, dx, dy, +3.0f);
                        }
                        // tilted x4
                        if (x + 4 * dx <= W && y + 4 * dx + dy <= H && x - dy >= 0) {
                            push2(true, x, y, 4 * dx, dy, -1.0f,
                                      x + dx, y + dx, 2 * dx, dy, +2.0f);
                        }
                        // tilted y4
                        if (x + dx <= W && y + dx + 4 * dy <= H && x - 4 * dy >= 0) {
                            push2(true, x, y, dx, 4 * dy, -1.0f,
                                      x - dy, y + dy, dx, 2 * dy, +2.0f);
                        }
                    }
                }
            }
        }
    }

    return out;
}

Status FaceVisionEngine::uploadFeaturePool(const std::vector<HaarFeature>& hFeatures, cudaStream_t stream) {
    if (hFeatures.empty()) {
        return Status::kInvalidArg;
    }

    if (dFeaturePool_) {
        cudaFree(dFeaturePool_);
        dFeaturePool_ = nullptr;
        featurePoolSize_ = 0;
    }

    const size_t bytes = hFeatures.size() * sizeof(HaarFeature);
    cudaError_t st = cudaMalloc(&dFeaturePool_, bytes);
    if (st != cudaSuccess) {
        return fromCuda(st);
    }

    st = cudaMemcpyAsync(dFeaturePool_, hFeatures.data(), bytes, cudaMemcpyHostToDevice, stream);
    if (st != cudaSuccess) {
        cudaFree(dFeaturePool_);
        dFeaturePool_ = nullptr;
        return fromCuda(st);
    }

    featurePoolSize_ = static_cast<int>(hFeatures.size());
    return Status::kOk;
}

Status FaceVisionEngine::computeIntegralBatchTransposed(const uint8_t* dGrayBatch,
                                                        int imageW,
                                                        int imageH,
                                                        int imageStride,
                                                        int nSamples,
                                                        cudaStream_t stream) {
    if (!dGrayBatch || imageW <= 0 || imageH <= 0 || imageStride < imageW || nSamples <= 0) {
        return Status::kInvalidArg;
    }
    if (imageW > maxImageW_ || imageH > maxImageH_ || nSamples > maxSamples_) {
        return Status::kInvalidArg;
    }

    currentImageW_ = imageW;
    currentImageH_ = imageH;
    currentSamples_ = nSamples;

    const int rowStride = maxImageW_ + 1;

    {
        const dim3 block(kScanThreads, 1, 1);
        const dim3 grid(static_cast<unsigned>(nSamples * imageH), 1, 1);
        RowPrefixKernel<kScanThreads><<<grid, block, 0, stream>>>(dGrayBatch,
                                                                   imageW,
                                                                   imageH,
                                                                   imageStride,
                                                                   nSamples,
                                                                   dRowPrefix_,
                                                                   dRowPrefixSq_,
                                                                   rowStride);
        const cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess) {
            return fromCuda(st);
        }
    }

    {
        if (nSamples == 1) {
            const int colThreads = 256;
            const dim3 block(static_cast<unsigned>(colThreads), 1, 1);
            const dim3 grid(static_cast<unsigned>(divUpHost(imageW + 1, colThreads)), 1, 1);
            ColumnScanTransposeSingleSampleKernel<<<grid, block, 0, stream>>>(dRowPrefix_,
                                                                                dRowPrefixSq_,
                                                                                imageW,
                                                                                imageH,
                                                                                rowStride,
                                                                                dSumT_,
                                                                                dSqsumT_);
        } else {
            const int colThreads = 128;
            const dim3 block(static_cast<unsigned>(colThreads), 1, 1);
            const dim3 grid(static_cast<unsigned>((nSamples + colThreads - 1) / colThreads),
                            static_cast<unsigned>(imageW + 1),
                            1);
            ColumnScanTransposeKernel<<<grid, block, 0, stream>>>(dRowPrefix_,
                                                                   dRowPrefixSq_,
                                                                   imageW,
                                                                   imageH,
                                                                   nSamples,
                                                                   rowStride,
                                                                   dSumT_,
                                                                   dSqsumT_);
        }
        const cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess) {
            return fromCuda(st);
        }
    }

    {
        const int iiWidth = imageW + 1;
        const int iiHeight = imageH + 1;
        const int normThreads = (nSamples < kNormThreads) ? nSamples : kNormThreads;
        const dim3 block(static_cast<unsigned>(normThreads), 1, 1);
        const dim3 grid(static_cast<unsigned>((nSamples + normThreads - 1) / normThreads), 1, 1);
        ComputeInvNormKernel<<<grid, block, 0, stream>>>(dSumT_, dSqsumT_, nSamples, iiWidth, iiHeight, dInvNorm_);
        const cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess) {
            return fromCuda(st);
        }
    }

    return Status::kOk;
}

IntegralImageSetT FaceVisionEngine::integralSetView() const noexcept {
    IntegralImageSetT out{};
    out.nSamples = currentSamples_;
    out.iiWidth = currentImageW_ + 1;
    out.iiHeight = currentImageH_ + 1;
    out.iiArea = out.iiWidth * out.iiHeight;
    out.d_sumT = dSumT_;
    out.d_sqsumT = dSqsumT_;
    out.d_tiltedT = nullptr;
    out.d_invNorm = dInvNorm_;
    return out;
}

Status FaceVisionEngine::setCascadeModel(const HaarFeature* dUsedFeatures,
                                         int usedFeatureCount,
                                         const GpuStump4* dStumps,
                                         int stumpCount,
                                         const std::vector<CascadeStage>& hStages,
                                         cudaStream_t stream) {
    if (!dUsedFeatures || !dStumps || hStages.empty() || usedFeatureCount <= 0 || stumpCount <= 0) {
        return Status::kInvalidArg;
    }

    auto ensureAlloc = [&](void** ptr, int neededCount, int& capacityCount, size_t elemSize) -> cudaError_t {
        if (neededCount <= 0) {
            return cudaErrorInvalidValue;
        }
        if (*ptr && capacityCount >= neededCount) {
            return cudaSuccess;
        }
        if (*ptr) {
            const cudaError_t freeSt = cudaFree(*ptr);
            if (freeSt != cudaSuccess) {
                return freeSt;
            }
            *ptr = nullptr;
            capacityCount = 0;
        }
        const cudaError_t allocSt = cudaMalloc(ptr, static_cast<size_t>(neededCount) * elemSize);
        if (allocSt == cudaSuccess) {
            capacityCount = neededCount;
        }
        return allocSt;
    };

    cudaError_t st = ensureAlloc(reinterpret_cast<void**>(&dModelFeatures_),
                                 usedFeatureCount,
                                 modelFeatureCapacity_,
                                 sizeof(HaarFeature));
    if (st != cudaSuccess) {
        clearCascadeModel(stream);
        return fromCuda(st);
    }
    st = ensureAlloc(reinterpret_cast<void**>(&dModelStumps_),
                     stumpCount,
                     modelStumpCapacity_,
                     sizeof(GpuStump4));
    if (st != cudaSuccess) {
        clearCascadeModel(stream);
        return fromCuda(st);
    }
    st = ensureAlloc(reinterpret_cast<void**>(&dModelStages_),
                     static_cast<int>(hStages.size()),
                     modelStageCapacity_,
                     sizeof(CascadeStage));
    if (st != cudaSuccess) {
        clearCascadeModel(stream);
        return fromCuda(st);
    }
    for (int r = 0; r < kLutRects; ++r) {
        for (int c = 0; c < kLutCorners; ++c) {
            st = ensureAlloc(reinterpret_cast<void**>(&dLutDx_[r][c]),
                             usedFeatureCount,
                             modelLutCapacity_,
                             sizeof(int32_t));
            if (st != cudaSuccess) {
                clearCascadeModel(stream);
                return fromCuda(st);
            }
            st = ensureAlloc(reinterpret_cast<void**>(&dLutDy_[r][c]),
                             usedFeatureCount,
                             modelLutCapacity_,
                             sizeof(int32_t));
            if (st != cudaSuccess) {
                clearCascadeModel(stream);
                return fromCuda(st);
            }
        }
        st = ensureAlloc(reinterpret_cast<void**>(&dLutW_[r]),
                         usedFeatureCount,
                         modelLutCapacity_,
                         sizeof(float));
        if (st != cudaSuccess) {
            clearCascadeModel(stream);
            return fromCuda(st);
        }
    }
    st = ensureAlloc(reinterpret_cast<void**>(&dLutRectCount_),
                     usedFeatureCount,
                     modelLutCapacity_,
                     sizeof(uint8_t));
    if (st != cudaSuccess) {
        clearCascadeModel(stream);
        return fromCuda(st);
    }

    st = cudaMemcpyAsync(dModelFeatures_, dUsedFeatures,
                         static_cast<size_t>(usedFeatureCount) * sizeof(HaarFeature),
                         cudaMemcpyDeviceToDevice, stream);
    if (st != cudaSuccess) {
        clearCascadeModel(stream);
        return fromCuda(st);
    }

    st = cudaMemcpyAsync(dModelStumps_, dStumps,
                         static_cast<size_t>(stumpCount) * sizeof(GpuStump4),
                         cudaMemcpyDeviceToDevice, stream);
    if (st != cudaSuccess) {
        clearCascadeModel(stream);
        return fromCuda(st);
    }

    st = cudaMemcpyAsync(dModelStages_, hStages.data(),
                         hStages.size() * sizeof(CascadeStage),
                         cudaMemcpyHostToDevice, stream);
    if (st != cudaSuccess) {
        clearCascadeModel(stream);
        return fromCuda(st);
    }

    {
        const dim3 block(256, 1, 1);
        const dim3 grid(static_cast<unsigned>((usedFeatureCount + 255) / 256), 1, 1);
        BuildFeatureLUTKernel<<<grid, block, 0, stream>>>(dModelFeatures_,
                                                           usedFeatureCount,
                                                           win_.winW + 1,
                                                           dLutDx_[0][0], dLutDy_[0][0],
                                                           dLutDx_[0][1], dLutDy_[0][1],
                                                           dLutDx_[0][2], dLutDy_[0][2],
                                                           dLutDx_[0][3], dLutDy_[0][3],
                                                           dLutW_[0],
                                                           dLutDx_[1][0], dLutDy_[1][0],
                                                           dLutDx_[1][1], dLutDy_[1][1],
                                                           dLutDx_[1][2], dLutDy_[1][2],
                                                           dLutDx_[1][3], dLutDy_[1][3],
                                                           dLutW_[1],
                                                           dLutDx_[2][0], dLutDy_[2][0],
                                                           dLutDx_[2][1], dLutDy_[2][1],
                                                           dLutDx_[2][2], dLutDy_[2][2],
                                                           dLutDx_[2][3], dLutDy_[2][3],
                                                           dLutW_[2],
                                                           dLutRectCount_);
        st = cudaGetLastError();
    if (st != cudaSuccess) {
        clearCascadeModel(stream);
        return fromCuda(st);
    }
    }

    modelFeatureCount_ = usedFeatureCount;
    modelStumpCount_ = stumpCount;
    modelStageCount_ = static_cast<int>(hStages.size());
    return Status::kOk;
}

Status FaceVisionEngine::clearCascadeModel(cudaStream_t) {
    if (dModelStages_) {
        cudaFree(dModelStages_);
        dModelStages_ = nullptr;
    }
    if (dModelStumps_) {
        cudaFree(dModelStumps_);
        dModelStumps_ = nullptr;
    }
    for (int r = 0; r < kLutRects; ++r) {
        for (int c = 0; c < kLutCorners; ++c) {
            if (dLutDx_[r][c]) {
                cudaFree(dLutDx_[r][c]);
                dLutDx_[r][c] = nullptr;
            }
            if (dLutDy_[r][c]) {
                cudaFree(dLutDy_[r][c]);
                dLutDy_[r][c] = nullptr;
            }
        }
        if (dLutW_[r]) {
            cudaFree(dLutW_[r]);
            dLutW_[r] = nullptr;
        }
    }
    if (dLutRectCount_) {
        cudaFree(dLutRectCount_);
        dLutRectCount_ = nullptr;
    }
    if (dModelFeatures_) {
        cudaFree(dModelFeatures_);
        dModelFeatures_ = nullptr;
    }
    modelFeatureCount_ = 0;
    modelFeatureCapacity_ = 0;
    modelLutCapacity_ = 0;
    modelStumpCount_ = 0;
    modelStumpCapacity_ = 0;
    modelStageCount_ = 0;
    modelStageCapacity_ = 0;
    return Status::kOk;
}

std::vector<Detection> FaceVisionEngine::groupRectanglesCpu(const std::vector<Detection>& in,
                                                            int minNeighbors,
                                                            float eps) const {
    if (in.empty()) {
        return {};
    }

    const int n = static_cast<int>(in.size());
    DisjointSet dsu(n);

    // OpenCV groupRectangles style clustering: merge similar rectangles.
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (similarRects(in[i], in[j], eps)) {
                dsu.unite(i, j);
            }
        }
    }

    std::vector<int64_t> sumX(n, 0), sumY(n, 0), sumW(n, 0), sumH(n, 0);
    std::vector<int> count(n, 0);
    std::vector<float> bestScore(n, -std::numeric_limits<float>::infinity());
    std::vector<int> scaleIdx(n, 0);

    for (int i = 0; i < n; ++i) {
        const int root = dsu.find(i);
        sumX[root] += in[i].x;
        sumY[root] += in[i].y;
        sumW[root] += in[i].w;
        sumH[root] += in[i].h;
        count[root] += 1;
        if (in[i].score > bestScore[root]) {
            bestScore[root] = in[i].score;
            scaleIdx[root] = in[i].scaleIdx;
        }
    }

    std::vector<Detection> clustered;
    clustered.reserve(n);
    std::vector<int> ccounts;
    ccounts.reserve(n);

    for (int i = 0; i < n; ++i) {
        if (dsu.find(i) != i || count[i] <= 0) {
            continue;
        }
        // Match OpenCV groupRectangles: need count > minNeighbors.
        if (count[i] <= std::max(1, minNeighbors)) {
            continue;
        }
        Detection d{};
        d.x = static_cast<int>(sumX[i] / count[i]);
        d.y = static_cast<int>(sumY[i] / count[i]);
        d.w = static_cast<int>(sumW[i] / count[i]);
        d.h = static_cast<int>(sumH[i] / count[i]);
        d.scaleIdx = scaleIdx[i];
        d.score = bestScore[i];
        clustered.push_back(d);
        ccounts.push_back(count[i]);
    }

    // Secondary suppression as in OpenCV: remove small rectangles inside larger ones.
    std::vector<Detection> out;
    for (size_t i = 0; i < clustered.size(); ++i) {
        const Detection& r1 = clustered[i];
        const int n1 = ccounts[i];
        bool discard = false;
        for (size_t j = 0; j < clustered.size(); ++j) {
            if (i == j) continue;
            const Detection& r2 = clustered[j];
            const int n2 = ccounts[j];

            const int dx = static_cast<int>(r2.w * eps);
            const int dy = static_cast<int>(r2.h * eps);

            if (r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.w <= r2.x + r2.w + dx &&
                r1.y + r1.h <= r2.y + r2.h + dy &&
                (n2 > std::max(3, n1) || n1 < 3)) {
                discard = true;
                break;
            }
        }
        if (!discard) out.push_back(r1);
    }

    return out;
}

Status FaceVisionEngine::detectMultiScale(const uint8_t* dImageGray,
                                          int imageW,
                                          int imageH,
                                          int imageStride,
                                          float scaleFactor,
                                          int minNeighbors,
                                          int minObjectSize,
                                          int maxDetections,
                                          std::vector<Detection>& outDetections,
                                          bool applyGrouping,
                                          cudaStream_t stream) {
    outDetections.clear();

    if (!dImageGray || imageW <= 0 || imageH <= 0 || imageStride < imageW ||
        scaleFactor <= 1.0f || minObjectSize < 1) {
        return Status::kInvalidArg;
    }
    if (!dModelFeatures_ || !dLutRectCount_ || !dLutDx_[0][0] || !dLutDy_[0][0] ||
        !dLutW_[0] || !dLutDx_[1][0] || !dLutDy_[1][0] || !dLutW_[1] ||
        !dModelStumps_ || !dModelStages_ || modelStageCount_ <= 0) {
        return Status::kInvalidArg;
    }

    const int globalCap = max(1, min(maxDetections, maxDetections_));

    {
        const int zero = 0;
        cudaError_t st = cudaMemcpyAsync(dDetectCount_, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (st != cudaSuccess) {
            return fromCuda(st);
        }
    }

    float scale = 1.0f;
    int scaleIdx = 0;

    while (true) {
        const int scaledW = static_cast<int>(std::round(static_cast<float>(imageW) / scale));
        const int scaledH = static_cast<int>(std::round(static_cast<float>(imageH) / scale));
        const int objW = static_cast<int>(std::round(static_cast<float>(win_.winW) * scale));
        const int objH = static_cast<int>(std::round(static_cast<float>(win_.winH) * scale));

        if (scaledW < win_.winW || scaledH < win_.winH) {
            break;
        }
        if (scaledW > maxImageW_ || scaledH > maxImageH_) {
            return Status::kInvalidArg;
        }
        if (objW < minObjectSize || objH < minObjectSize) {
            scale *= scaleFactor;
            ++scaleIdx;
            continue;
        }

        const uint8_t* dScaled = nullptr;
        int scaledStride = 0;

        if (scaleIdx == 0) {
            dScaled = dImageGray;
            scaledStride = imageStride;
        } else {
            dScaled = dPyrBufA_;
            scaledStride = scaledW;
            const dim3 block(16, 16, 1);
            const dim3 grid(static_cast<unsigned>(divUpHost(scaledW, 16)),
                            static_cast<unsigned>(divUpHost(scaledH, 16)),
                            1);
            ResizeBilinearKernel<<<grid, block, 0, stream>>>(dImageGray,
                                                              imageW,
                                                              imageH,
                                                              imageStride,
                                                              dPyrBufA_,
                                                              scaledW,
                                                              scaledH,
                                                              scaledStride,
                                                              scale);
            cudaError_t st = cudaGetLastError();
            if (st != cudaSuccess) {
                return fromCuda(st);
            }
        }

        Status s = computeIntegralBatchTransposed(dScaled, scaledW, scaledH, scaledStride, 1, stream);
        if (s != Status::kOk) {
            return s;
        }

        const int workW = scaledW - win_.winW + 1;
        const int workH = scaledH - win_.winH + 1;
        // Scale-aware jump scanning: skip more positions at large scales.
        const int step = (scale >= 2.0f) ? 2 : 1;
        const int scanW = divUpHost(workW, step);
        const int scanH = divUpHost(workH, step);

        if (scanW > 0 && scanH > 0) {
            // Keep x dimension at warp width to improve memory coalescing on integral reads.
            const dim3 block(32, 8, 1);
            const dim3 grid(static_cast<unsigned>(divUpHost(scanW, 32)),
                            static_cast<unsigned>(divUpHost(scanH, 8)),
                            1);

            DetectCascadeKernel<<<grid, block, 0, stream>>>(dSumT_,
                                                             dSqsumT_,
                                                             scaledW + 1,
                                                             scaledH + 1,
                                                             dLutDx_[0][0], dLutDy_[0][0],
                                                             dLutDx_[0][1], dLutDy_[0][1],
                                                             dLutDx_[0][2], dLutDy_[0][2],
                                                             dLutDx_[0][3], dLutDy_[0][3],
                                                             dLutW_[0],
                                                             dLutDx_[1][0], dLutDy_[1][0],
                                                             dLutDx_[1][1], dLutDy_[1][1],
                                                             dLutDx_[1][2], dLutDy_[1][2],
                                                             dLutDx_[1][3], dLutDy_[1][3],
                                                             dLutW_[1],
                                                             dLutDx_[2][0], dLutDy_[2][0],
                                                             dLutDx_[2][1], dLutDy_[2][1],
                                                             dLutDx_[2][2], dLutDy_[2][2],
                                                             dLutDx_[2][3], dLutDy_[2][3],
                                                             dLutW_[2],
                                                             dLutRectCount_,
                                                             dModelStumps_,
                                                             dModelStages_,
                                                             modelStageCount_,
                                                             win_.winW,
                                                             win_.winH,
                                                             scaleIdx,
                                                             scale,
                                                             step,
                                                             dDetectOut_,
                                                             dDetectCount_,
                                                             globalCap);
            cudaError_t st = cudaGetLastError();
            if (st != cudaSuccess) {
                return fromCuda(st);
            }
        }

        scale *= scaleFactor;
        ++scaleIdx;
    }

    cudaError_t stc = cudaMemcpyAsync(hDetectCountPinned_, dDetectCount_, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (stc != cudaSuccess) {
        return fromCuda(stc);
    }
    stc = cudaMemcpyAsync(hDetectPinned_,
                          dDetectOut_,
                          static_cast<size_t>(globalCap) * sizeof(Detection),
                          cudaMemcpyDeviceToHost,
                          stream);
    if (stc != cudaSuccess) {
        return fromCuda(stc);
    }
    stc = cudaStreamSynchronize(stream);
    if (stc != cudaSuccess) {
        return fromCuda(stc);
    }
    int hCount = *hDetectCountPinned_;
    hCount = min(hCount, globalCap);

    std::vector<Detection> allDetections;
    if (hCount > 0) {
        allDetections.assign(hDetectPinned_, hDetectPinned_ + hCount);
    }

    if (applyGrouping) {
        outDetections = groupRectanglesCpu(allDetections, minNeighbors, 0.2f);
    } else {
        outDetections.swap(allDetections);
    }
    return Status::kOk;
}

} // namespace vj
