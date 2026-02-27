#include "vj/boosting_core.cuh"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <cub/cub.cuh>

namespace vj {
namespace {

constexpr int kEvalThreads = 256;
constexpr int kThresholdThreads = 256;
constexpr int kSortThreads = 256;
constexpr float kValueEps = 1e-7f;

template <typename T>
__device__ __forceinline__ T ldgRead(const T* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(p);
#else
    return *p;
#endif
}

struct LocalBest {
    float err;
    int split;
    int parity;
};

struct TileBestLocal {
    float err;
    float theta;
    int featureIdx;
    int parity;
};

__device__ __forceinline__ LocalBest betterBest(const LocalBest& a, const LocalBest& b) {
    if (b.err < a.err) {
        return b;
    }
    if (b.err > a.err) {
        return a;
    }
    // Tie-break for determinism: smaller split first, then +1 parity first.
    if (b.split < a.split) {
        return b;
    }
    if (b.split > a.split) {
        return a;
    }
    if (b.parity > a.parity) {
        return b;
    }
    return a;
}

__device__ __forceinline__ TileBestLocal betterTileBest(const TileBestLocal& a, const TileBestLocal& b) {
    if (b.err < a.err) {
        return b;
    }
    if (b.err > a.err) {
        return a;
    }
    if (b.featureIdx < a.featureIdx) {
        return b;
    }
    return a;
}

__global__ void InitSortValueIndicesKernel(uint16_t* __restrict__ d_vals,
                                           int nSamples) {
    const int sample = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int feature = static_cast<int>(blockIdx.y);
    if (sample >= nSamples) {
        return;
    }
    d_vals[static_cast<size_t>(feature) * nSamples + sample] = static_cast<uint16_t>(sample);
}

__global__ void FillSegmentOffsetsKernel(int* __restrict__ d_offsets,
                                         int featureCount,
                                         int nSamples) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i > featureCount) {
        return;
    }
    d_offsets[i] = i * nSamples;
}

__global__ void SelectBestInTileKernel(const FeatureBestSplit* __restrict__ d_best,
                                       int featureBegin,
                                       int featureCount,
                                       const uint8_t* __restrict__ d_usedMask,
                                       TileBestCandidate* __restrict__ d_out) {
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ TileBestLocal sBest[kThresholdThreads];

    TileBestLocal local{FLT_MAX, 0.0f, INT_MAX, +1};
    for (int i = tid; i < featureCount; i += blockDim.x) {
        const int featureIdx = featureBegin + i;
        if (d_usedMask && d_usedMask[featureIdx] != 0) {
            continue;
        }
        const FeatureBestSplit b = d_best[i];
        TileBestLocal cand{b.bestErr, b.bestTheta, featureIdx, static_cast<int>(b.bestParity)};
        local = betterTileBest(local, cand);
    }
    sBest[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sBest[tid] = betterTileBest(sBest[tid], sBest[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        TileBestCandidate out{};
        out.bestErr = sBest[0].err;
        out.bestTheta = sBest[0].theta;
        out.featureIdx = sBest[0].featureIdx;
        out.bestParity = static_cast<int8_t>(sBest[0].parity);
        d_out[0] = out;
    }
}

// -----------------------------------------------------------------------------
// Core-1: Feature response evaluation kernel
// Grid mapping:
//   blockIdx.x -> feature index inside tile
//   blockIdx.y + threadIdx.x -> sample index
//
// Important memory design:
//   sumT is [iiArea][N]. For any fixed offset p, samples are contiguous.
//   Threads in a warp read sumT[p*N + sample], so memory access is coalesced.
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2) void EvaluateFeatureResponsesKernel(const HaarFeature* __restrict__ d_features,
                                                                          int featureBegin,
                                                                          int featureCount,
                                                                          const int32_t* __restrict__ d_sumT,
                                                                          const float* __restrict__ d_invNorm,
                                                                          int nSamples,
                                                                          float* __restrict__ d_respOut) {
    const int featureLocal = blockIdx.x;
    const int sample = blockIdx.y * blockDim.x + threadIdx.x;

    if (featureLocal >= featureCount || sample >= nSamples) {
        return;
    }

    __shared__ HaarFeature f;
    if (threadIdx.x == 0) {
        f = d_features[featureBegin + featureLocal];
    }
    __syncthreads();

    // Each load pattern below is coalesced across threads for fixed offset p*.
    const auto loadSumT = [&](int p) -> float {
        return static_cast<float>(ldgRead(d_sumT + static_cast<size_t>(p) * nSamples + sample));
    };

    const float s0 = loadSumT(f.r0.p0) - loadSumT(f.r0.p1) - loadSumT(f.r0.p2) + loadSumT(f.r0.p3);
    const float s1 = loadSumT(f.r1.p0) - loadSumT(f.r1.p1) - loadSumT(f.r1.p2) + loadSumT(f.r1.p3);

    float resp = f.r0.w * s0 + f.r1.w * s1;
    if (f.rectCount == 3) {
        const float s2 = loadSumT(f.r2.p0) - loadSumT(f.r2.p1) - loadSumT(f.r2.p2) + loadSumT(f.r2.p3);
        resp += f.r2.w * s2;
    }

    const float invNorm = d_invNorm ? ldgRead(d_invNorm + sample) : 1.0f;
    d_respOut[static_cast<size_t>(featureLocal) * nSamples + sample] = resp * invNorm;
}

// -----------------------------------------------------------------------------
// Core-2: Threshold sweep kernel (1 block = 1 feature)
//
// This kernel does three things fully on GPU:
//   1) Reduce total positive/negative weight for the feature.
//   2) Sweep sorted samples with shared-memory prefix scan to evaluate split errors.
//   3) Block min-reduction to get best (err, split, parity), then output theta.
// -----------------------------------------------------------------------------
__global__ void EvaluateAndFindThresholdKernel(const float* __restrict__ d_resp,
                                               const uint16_t* __restrict__ d_sortedIdx,
                                               const int8_t* __restrict__ d_label,
                                               const uint8_t* __restrict__ d_active,
                                               const float* __restrict__ d_weight,
                                               int nSamples,
                                               int featureCount,
                                               FeatureBestSplit* __restrict__ d_outBest) {
    const int tid = threadIdx.x;
    const int featureLocal = blockIdx.x;

    if (featureLocal >= featureCount) {
        return;
    }

    const float* respRow = d_resp + static_cast<size_t>(featureLocal) * nSamples;
    const uint16_t* ord = d_sortedIdx + static_cast<size_t>(featureLocal) * nSamples;

    __shared__ float sPos[kThresholdThreads];
    __shared__ float sNeg[kThresholdThreads];
    __shared__ float sErr[kThresholdThreads];
    __shared__ int sSplit[kThresholdThreads];
    __shared__ int sParity[kThresholdThreads];

    __shared__ float sTotalPos;
    __shared__ float sTotalNeg;
    __shared__ float sCarryPos;
    __shared__ float sCarryNeg;

    // --------------------
    // Step A: global totals
    // --------------------
    float localPos = 0.0f;
    float localNeg = 0.0f;

    for (int k = tid; k < nSamples; k += blockDim.x) {
        const int si = static_cast<int>(ord[k]);
        const bool active = (d_active == nullptr) || (d_active[si] != 0);
        if (!active) {
            continue;
        }
        const float w = d_weight[si];
        if (d_label[si] > 0) {
            localPos += w;
        } else {
            localNeg += w;
        }
    }

    sPos[tid] = localPos;
    sNeg[tid] = localNeg;
    __syncthreads();

    // Block reduction for totals.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sPos[tid] += sPos[tid + stride];
            sNeg[tid] += sNeg[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sTotalPos = sPos[0];
        sTotalNeg = sNeg[0];
        sCarryPos = 0.0f;
        sCarryNeg = 0.0f;
    }
    __syncthreads();

    // Thread-local best candidate.
    LocalBest best{FLT_MAX, -1, +1};

    // ---------------------------------------------------------
    // Step B: tile-by-tile prefix scan over sorted sample order
    // ---------------------------------------------------------
    for (int base = 0; base < nSamples; base += blockDim.x) {
        const int k = base + tid;

        float wp = 0.0f;
        float wn = 0.0f;
        if (k < nSamples) {
            const int si = static_cast<int>(ord[k]);
            const bool active = (d_active == nullptr) || (d_active[si] != 0);
            if (active) {
                const float w = d_weight[si];
                if (d_label[si] > 0) {
                    wp = w;
                } else {
                    wn = w;
                }
            }
        }

        sPos[tid] = wp;
        sNeg[tid] = wn;
        __syncthreads();

        // Shared-memory inclusive prefix scan.
        // 中文: 每轮 offset 翻倍，先读旧值再同步，再写回，避免读写冲突。
        // EN: Doubling-offset inclusive scan with synchronized read/modify/write.
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            float addPos = 0.0f;
            float addNeg = 0.0f;
            if (tid >= offset) {
                addPos = sPos[tid - offset];
                addNeg = sNeg[tid - offset];
            }
            __syncthreads();
            if (tid >= offset) {
                sPos[tid] += addPos;
                sNeg[tid] += addNeg;
            }
            __syncthreads();
        }

        const float carryPos = sCarryPos;
        const float carryNeg = sCarryNeg;

        // Evaluate split at k (between k and k+1).
        if (k < nSamples - 1) {
            const int si0 = static_cast<int>(ord[k]);
            const int si1 = static_cast<int>(ord[k + 1]);
            const float v0 = respRow[si0];
            const float v1 = respRow[si1];

            // Only consider valid threshold between distinct sorted values.
            if (v0 + kValueEps < v1) {
                const float wPosLeft = carryPos + sPos[tid];
                const float wNegLeft = carryNeg + sNeg[tid];
                const float wPosRight = sTotalPos - wPosLeft;
                const float wNegRight = sTotalNeg - wNegLeft;

                // left predicts +1, right predicts -1
                // errors = neg_left + pos_right
                const float errP = wNegLeft + wPosRight;
                LocalBest candP{errP, k, -1};
                best = betterBest(best, candP);

                // left predicts -1, right predicts +1
                // errors = pos_left + neg_right
                const float errN = wPosLeft + wNegRight;
                LocalBest candN{errN, k, +1};
                best = betterBest(best, candN);
            }
        }

        // Update carries (cumulative left-side sums) once per tile.
        if (tid == 0) {
            const int validCount = min(blockDim.x, nSamples - base);
            if (validCount > 0) {
                sCarryPos += sPos[validCount - 1];
                sCarryNeg += sNeg[validCount - 1];
            }
        }
        __syncthreads();
    }

    // -------------------------------------
    // Step C: block min reduction for best.
    // -------------------------------------
    sErr[tid] = best.err;
    sSplit[tid] = best.split;
    sParity[tid] = best.parity;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            LocalBest a{sErr[tid], sSplit[tid], sParity[tid]};
            LocalBest b{sErr[tid + stride], sSplit[tid + stride], sParity[tid + stride]};
            const LocalBest c = betterBest(a, b);
            sErr[tid] = c.err;
            sSplit[tid] = c.split;
            sParity[tid] = c.parity;
        }
        __syncthreads();
    }

    if (tid == 0) {
        FeatureBestSplit out{};
        out.bestErr = sErr[0];
        out.bestSplitPos = sSplit[0];
        out.bestParity = static_cast<int8_t>(sParity[0]);

        if (out.bestSplitPos >= 0 && out.bestSplitPos < nSamples - 1) {
            const int si0 = static_cast<int>(ord[out.bestSplitPos]);
            const int si1 = static_cast<int>(ord[out.bestSplitPos + 1]);
            out.bestTheta = 0.5f * (respRow[si0] + respRow[si1]);
        } else {
            out.bestTheta = 0.0f;
            out.bestErr = FLT_MAX;
            out.bestParity = +1;
        }

        d_outBest[featureLocal] = out;
    }
}

} // namespace

Status AdaBoostTrainer::fromCuda(cudaError_t err) {
    return err == cudaSuccess ? Status::kOk : Status::kCudaError;
}

AdaBoostTrainer::AdaBoostTrainer(const WindowSpec& win, int maxSamples, int maxFeaturesPerTile)
    : win_(win), maxSamples_(maxSamples), maxFeaturesPerTile_(maxFeaturesPerTile) {
    if (maxSamples_ <= 0 || maxFeaturesPerTile_ <= 0) {
        throw std::invalid_argument("AdaBoostTrainer: invalid maxSamples/maxFeaturesPerTile");
    }

    const size_t respCount = static_cast<size_t>(maxSamples_) * static_cast<size_t>(maxFeaturesPerTile_);
    if (respCount > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument("AdaBoostTrainer: maxSamples*maxFeaturesPerTile exceeds int range");
    }
    const size_t respBytes = respCount * sizeof(float);
    const size_t bestBytes = static_cast<size_t>(maxFeaturesPerTile_) * sizeof(FeatureBestSplit);
    const size_t tileBestBytes = sizeof(TileBestCandidate);
    const size_t idxBytes = respCount * sizeof(uint16_t);
    const size_t segOfsBytes = static_cast<size_t>(maxFeaturesPerTile_ + 1) * sizeof(int);

    auto alloc_or_throw = [](void** ptr, size_t bytes, const char* msg) {
        const cudaError_t st = cudaMalloc(ptr, bytes);
        if (st != cudaSuccess) {
            throw std::runtime_error(msg);
        }
    };

    try {
        alloc_or_throw(reinterpret_cast<void**>(&dResponseBuffer_), respBytes,
                       "AdaBoostTrainer: cudaMalloc dResponseBuffer_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dBestBuffer_), bestBytes,
                       "AdaBoostTrainer: cudaMalloc dBestBuffer_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dTileBestBuffer_), tileBestBytes,
                       "AdaBoostTrainer: cudaMalloc dTileBestBuffer_ failed");

        alloc_or_throw(reinterpret_cast<void**>(&dSortKeysOut_), respBytes,
                       "AdaBoostTrainer: cudaMalloc dSortKeysOut_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dSortValsIn_), idxBytes,
                       "AdaBoostTrainer: cudaMalloc dSortValsIn_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dSortValsOut_), idxBytes,
                       "AdaBoostTrainer: cudaMalloc dSortValsOut_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dSegmentOffsets_), segOfsBytes,
                       "AdaBoostTrainer: cudaMalloc dSegmentOffsets_ failed");

        sortTempStorageBytes_ = 0;
        cudaError_t st = cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr,
            sortTempStorageBytes_,
            dResponseBuffer_,
            dSortKeysOut_,
            dSortValsIn_,
            dSortValsOut_,
            static_cast<int>(respCount),
            maxFeaturesPerTile_,
            dSegmentOffsets_,
            dSegmentOffsets_ + 1,
            0,
            static_cast<int>(8 * sizeof(float)));
        if (st != cudaSuccess) {
            throw std::runtime_error("AdaBoostTrainer: CUB temp storage query failed");
        }

        if (sortTempStorageBytes_ > 0) {
            alloc_or_throw(&dSortTempStorage_, sortTempStorageBytes_,
                           "AdaBoostTrainer: cudaMalloc dSortTempStorage_ failed");
        }
    } catch (...) {
        if (dSortTempStorage_) cudaFree(dSortTempStorage_);
        if (dSegmentOffsets_) cudaFree(dSegmentOffsets_);
        if (dSortValsOut_) cudaFree(dSortValsOut_);
        if (dSortValsIn_) cudaFree(dSortValsIn_);
        if (dSortKeysOut_) cudaFree(dSortKeysOut_);
        if (dTileBestBuffer_) cudaFree(dTileBestBuffer_);
        if (dBestBuffer_) cudaFree(dBestBuffer_);
        if (dResponseBuffer_) cudaFree(dResponseBuffer_);
        dSortTempStorage_ = nullptr;
        dSegmentOffsets_ = nullptr;
        dSortValsOut_ = nullptr;
        dSortValsIn_ = nullptr;
        dSortKeysOut_ = nullptr;
        dTileBestBuffer_ = nullptr;
        dBestBuffer_ = nullptr;
        dResponseBuffer_ = nullptr;
        throw;
    }
}

AdaBoostTrainer::~AdaBoostTrainer() {
    if (dSortTempStorage_) {
        cudaFree(dSortTempStorage_);
        dSortTempStorage_ = nullptr;
    }
    if (dSegmentOffsets_) {
        cudaFree(dSegmentOffsets_);
        dSegmentOffsets_ = nullptr;
    }
    if (dSortValsOut_) {
        cudaFree(dSortValsOut_);
        dSortValsOut_ = nullptr;
    }
    if (dSortValsIn_) {
        cudaFree(dSortValsIn_);
        dSortValsIn_ = nullptr;
    }
    if (dSortKeysOut_) {
        cudaFree(dSortKeysOut_);
        dSortKeysOut_ = nullptr;
    }
    if (dTileBestBuffer_) {
        cudaFree(dTileBestBuffer_);
        dTileBestBuffer_ = nullptr;
    }
    if (dResponseBuffer_) {
        cudaFree(dResponseBuffer_);
        dResponseBuffer_ = nullptr;
    }
    if (dBestBuffer_) {
        cudaFree(dBestBuffer_);
        dBestBuffer_ = nullptr;
    }
}

AdaBoostTrainer::AdaBoostTrainer(AdaBoostTrainer&& other) noexcept {
    *this = std::move(other);
}

AdaBoostTrainer& AdaBoostTrainer::operator=(AdaBoostTrainer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (dResponseBuffer_) {
        cudaFree(dResponseBuffer_);
        dResponseBuffer_ = nullptr;
    }
    if (dBestBuffer_) {
        cudaFree(dBestBuffer_);
        dBestBuffer_ = nullptr;
    }
    if (dSortTempStorage_) {
        cudaFree(dSortTempStorage_);
        dSortTempStorage_ = nullptr;
    }
    if (dSegmentOffsets_) {
        cudaFree(dSegmentOffsets_);
        dSegmentOffsets_ = nullptr;
    }
    if (dSortValsOut_) {
        cudaFree(dSortValsOut_);
        dSortValsOut_ = nullptr;
    }
    if (dSortValsIn_) {
        cudaFree(dSortValsIn_);
        dSortValsIn_ = nullptr;
    }
    if (dSortKeysOut_) {
        cudaFree(dSortKeysOut_);
        dSortKeysOut_ = nullptr;
    }
    if (dTileBestBuffer_) {
        cudaFree(dTileBestBuffer_);
        dTileBestBuffer_ = nullptr;
    }

    win_ = other.win_;
    maxSamples_ = other.maxSamples_;
    maxFeaturesPerTile_ = other.maxFeaturesPerTile_;
    iiSet_ = other.iiSet_;
    labels_ = other.labels_;
    weights_ = other.weights_;
    dResponseBuffer_ = other.dResponseBuffer_;
    dBestBuffer_ = other.dBestBuffer_;
    dSortKeysOut_ = other.dSortKeysOut_;
    dSortValsIn_ = other.dSortValsIn_;
    dSortValsOut_ = other.dSortValsOut_;
    dSegmentOffsets_ = other.dSegmentOffsets_;
    dSortTempStorage_ = other.dSortTempStorage_;
    sortTempStorageBytes_ = other.sortTempStorageBytes_;
    dTileBestBuffer_ = other.dTileBestBuffer_;
    sortLayoutSamples_ = other.sortLayoutSamples_;
    sortLayoutReady_ = other.sortLayoutReady_;

    other.maxSamples_ = 0;
    other.maxFeaturesPerTile_ = 0;
    other.iiSet_ = IntegralImageSetT{};
    other.labels_ = LabelsView{};
    other.weights_ = WeightsView{};
    other.dResponseBuffer_ = nullptr;
    other.dBestBuffer_ = nullptr;
    other.dSortKeysOut_ = nullptr;
    other.dSortValsIn_ = nullptr;
    other.dSortValsOut_ = nullptr;
    other.dSegmentOffsets_ = nullptr;
    other.dSortTempStorage_ = nullptr;
    other.dTileBestBuffer_ = nullptr;
    other.sortTempStorageBytes_ = 0;
    other.sortLayoutSamples_ = 0;
    other.sortLayoutReady_ = false;
    return *this;
}

Status AdaBoostTrainer::bindTrainingSet(const IntegralImageSetT& iiSet,
                                        const LabelsView& labels,
                                        const WeightsView& weights) {
    if (!iiSet.d_sumT || !iiSet.d_invNorm || !labels.d_label || !weights.d_weight) {
        return Status::kInvalidArg;
    }
    if (iiSet.nSamples <= 0 || iiSet.nSamples > maxSamples_) {
        return Status::kInvalidArg;
    }
    // sorted index buffer in this core uses uint16_t
    if (iiSet.nSamples > 65535) {
        return Status::kInvalidArg;
    }
    if (labels.nSamples != iiSet.nSamples || weights.nSamples != iiSet.nSamples) {
        return Status::kInvalidArg;
    }

    iiSet_ = iiSet;
    labels_ = labels;
    weights_ = weights;

    if (!sortLayoutReady_ || sortLayoutSamples_ != iiSet.nSamples) {
        const int nSamples = iiSet.nSamples;
        {
            const dim3 block(kSortThreads, 1, 1);
            const dim3 grid(static_cast<unsigned>((nSamples + kSortThreads - 1) / kSortThreads),
                            static_cast<unsigned>(maxFeaturesPerTile_),
                            1);
            InitSortValueIndicesKernel<<<grid, block>>>(dSortValsIn_, nSamples);
            const cudaError_t st = cudaGetLastError();
            if (st != cudaSuccess) {
                return fromCuda(st);
            }
        }
        {
            const dim3 block(kSortThreads, 1, 1);
            const int n = maxFeaturesPerTile_ + 1;
            const dim3 grid(static_cast<unsigned>((n + kSortThreads - 1) / kSortThreads), 1, 1);
            FillSegmentOffsetsKernel<<<grid, block>>>(dSegmentOffsets_, maxFeaturesPerTile_, nSamples);
            const cudaError_t st = cudaGetLastError();
            if (st != cudaSuccess) {
                return fromCuda(st);
            }
        }
        sortLayoutSamples_ = nSamples;
        sortLayoutReady_ = true;
    }
    return Status::kOk;
}

Status AdaBoostTrainer::evaluateFeatureResponses(const HaarFeature* dFeatures,
                                                 int featureBegin,
                                                 int featureCount,
                                                 float* dOutResp,
                                                 cudaStream_t stream) const {
    if (!dFeatures || !iiSet_.d_sumT || !iiSet_.d_invNorm) {
        return Status::kInvalidArg;
    }
    if (featureBegin < 0 || featureCount <= 0 || featureCount > maxFeaturesPerTile_) {
        return Status::kInvalidArg;
    }

    const int nSamples = iiSet_.nSamples;
    float* out = dOutResp ? dOutResp : dResponseBuffer_;
    if (!out) {
        return Status::kInvalidArg;
    }

    const dim3 block(kEvalThreads, 1, 1);
    const dim3 grid(static_cast<unsigned>(featureCount),
                    static_cast<unsigned>((nSamples + kEvalThreads - 1) / kEvalThreads),
                    1);

    EvaluateFeatureResponsesKernel<<<grid, block, 0, stream>>>(dFeatures,
                                                                featureBegin,
                                                                featureCount,
                                                                iiSet_.d_sumT,
                                                                iiSet_.d_invNorm,
                                                                nSamples,
                                                                out);
    return fromCuda(cudaGetLastError());
}

Status AdaBoostTrainer::evaluateAndFindThreshold(const FeatureTileView& tile,
                                                 FeatureBestSplit* dOutBestPerFeature,
                                                 cudaStream_t stream) const {
    if (!tile.d_resp || !tile.d_sortedIdx || !labels_.d_label || !weights_.d_weight) {
        return Status::kInvalidArg;
    }
    if (tile.featureCount <= 0 || tile.featureCount > maxFeaturesPerTile_) {
        return Status::kInvalidArg;
    }

    const int nSamples = iiSet_.nSamples;
    if (nSamples <= 1) {
        return Status::kInvalidArg;
    }

    FeatureBestSplit* out = dOutBestPerFeature ? dOutBestPerFeature : dBestBuffer_;
    if (!out) {
        return Status::kInvalidArg;
    }

    const dim3 block(kThresholdThreads, 1, 1);
    const dim3 grid(static_cast<unsigned>(tile.featureCount), 1, 1);

    EvaluateAndFindThresholdKernel<<<grid, block, 0, stream>>>(tile.d_resp,
                                                                tile.d_sortedIdx,
                                                                labels_.d_label,
                                                                labels_.d_active,
                                                                weights_.d_weight,
                                                                nSamples,
                                                                tile.featureCount,
                                                                out);
    return fromCuda(cudaGetLastError());
}

Status AdaBoostTrainer::sortSamplesPerFeature(const float* dResp,
                                              int featureCount,
                                              uint16_t* dOutSortedIdx,
                                              cudaStream_t stream) {
    if (!dResp || !dSortValsIn_ || !dSortValsOut_ || !dSortKeysOut_ || !dSegmentOffsets_) {
        return Status::kInvalidArg;
    }
    if (featureCount <= 0 || featureCount > maxFeaturesPerTile_) {
        return Status::kInvalidArg;
    }

    const int nSamples = iiSet_.nSamples;
    if (nSamples <= 0 || nSamples > 65535) {
        return Status::kInvalidArg;
    }

    const int totalItems = featureCount * nSamples;

    if (!sortLayoutReady_ || sortLayoutSamples_ != nSamples) {
        return Status::kInvalidArg;
    }

    // Sort all feature segments in one CUB call.
    const cudaError_t sortSt = cub::DeviceSegmentedRadixSort::SortPairs(
        dSortTempStorage_,
        sortTempStorageBytes_,
        dResp,               // keys in: feature responses
        dSortKeysOut_,       // keys out: sorted responses (kept for debug/optional reuse)
        dSortValsIn_,        // values in: sample indices
        dSortValsOut_,       // values out: sorted sample indices
        totalItems,
        featureCount,
        dSegmentOffsets_,
        dSegmentOffsets_ + 1,
        0,
        static_cast<int>(8 * sizeof(float)),
        stream);
    if (sortSt != cudaSuccess) {
        return fromCuda(sortSt);
    }

    // Optionally export to caller-provided output buffer.
    if (dOutSortedIdx && dOutSortedIdx != dSortValsOut_) {
        const cudaError_t copySt = cudaMemcpyAsync(
            dOutSortedIdx,
            dSortValsOut_,
            static_cast<size_t>(totalItems) * sizeof(uint16_t),
            cudaMemcpyDeviceToDevice,
            stream);
        return fromCuda(copySt);
    }
    return Status::kOk;
}

Status AdaBoostTrainer::selectBestInTile(int featureBegin,
                                         int featureCount,
                                         const uint8_t* dUsedFeatureMask,
                                         TileBestCandidate* dOut,
                                         cudaStream_t stream) const {
    if (featureBegin < 0 || featureCount <= 0 || featureCount > maxFeaturesPerTile_) {
        return Status::kInvalidArg;
    }
    if (!dBestBuffer_) {
        return Status::kInvalidArg;
    }
    TileBestCandidate* out = dOut ? dOut : dTileBestBuffer_;
    if (!out) {
        return Status::kInvalidArg;
    }

    const dim3 block(kThresholdThreads, 1, 1);
    const dim3 grid(1, 1, 1);
    SelectBestInTileKernel<<<grid, block, 0, stream>>>(dBestBuffer_,
                                                        featureBegin,
                                                        featureCount,
                                                        dUsedFeatureMask,
                                                        out);
    return fromCuda(cudaGetLastError());
}

} // namespace vj
