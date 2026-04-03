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

// 这个文件专门负责 AdaBoost 训练里“找当前最优弱分类器”的 GPU 部分。
// 整体思路可以概括成 4 步：
// 1) 对一批 Haar 特征计算所有样本上的响应值；
// 2) 对每个特征自己的响应序列做排序；
// 3) 在排好序的响应上扫描所有可能阈值，找最小分类误差；
// 4) 从一个 tile 的所有特征里再挑出整体最优候选。
//
// 之所以要这样拆，是因为 AdaBoost 每一轮都要重复“遍历所有候选特征找最优 stump”，
// 而这一步恰好是最耗时的热点，所以非常适合搬到 GPU。
constexpr int kEvalThreads = 256;
constexpr int kThresholdThreads = 256;
constexpr int kSortThreads = 256;
// 浮点比较时留一点 epsilon，避免两个几乎相等的响应因为数值误差被误判成可分裂点。
constexpr float kValueEps = 1e-7f;

template <typename T>
__device__ __forceinline__ T ldgRead(const T* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    // 对只读数据尽量走只读缓存路径，降低普通 global load 的压力。
    return __ldg(p);
#else
    return *p;
#endif
}

struct LocalBest {
    // 当前线程/当前归约阶段看到的最优误差。
    float err;
    // split 表示阈值放在排好序后的第几个样本与下一个样本之间。
    int split;
    // parity 决定阈值左边判成 +1 还是 -1。
    int parity;
};

struct WeightPair {
    // AdaBoost 训练的核心是“加权错误率”，所以这里分别累计正负样本权重。
    float pos;
    float neg;
};

struct TileBestLocal {
    // 一个 feature tile 内的最佳候选，用于第二层归约。
    float err;
    float theta;
    int featureIdx;
    int parity;
};

__device__ __forceinline__ WeightPair operator+(const WeightPair& a, const WeightPair& b) {
    // 为 CUB 的 BlockReduce 提供“两个局部统计量怎么合并”的规则。
    return WeightPair{a.pos + b.pos, a.neg + b.neg};
}

__device__ __forceinline__ LocalBest betterBest(const LocalBest& a, const LocalBest& b) {
    // 这个比较器定义了“哪个 split 更好”。
    // 主排序键是误差；平手时再用 split 和 parity 打破平局，保证结果稳定可复现。
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

struct BetterBestOp {
    __device__ __forceinline__ LocalBest operator()(const LocalBest& a, const LocalBest& b) const {
        return betterBest(a, b);
    }
};

__device__ __forceinline__ TileBestLocal betterTileBest(const TileBestLocal& a, const TileBestLocal& b) {
    // tile 级比较器：先比误差，再比特征下标，保证同样输入下结果 deterministic。
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

// 输入:
//   d_vals: [featureCount][nSamples] 的 value 数组缓冲区。
//   nSamples: 当前训练样本数。
//
// 输出:
//   对每个特征行写入 0..nSamples-1 的样本编号。
//
// 作用:
//   为后续“按响应排序”准备 value 数组。排序完成后，value 会变成“有序样本下标”。
__global__ void InitSortValueIndicesKernel(uint16_t* __restrict__ d_vals,
                                           int nSamples) {
    const int sample = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int feature = static_cast<int>(blockIdx.y);
    if (sample >= nSamples) {
        return;
    }
    // CUB 的分段排序是“key/value 一起排”：
    // key 是响应值，value 放原始 sample 下标，这样排序后我们还能知道样本原次序。
    d_vals[static_cast<size_t>(feature) * nSamples + sample] = static_cast<uint16_t>(sample);
}

// 输入:
//   d_offsets: 长度为 featureCount + 1 的 offset 缓冲区。
//   featureCount: 当前 tile 中的特征数。
//   nSamples: 当前训练样本数。
//
// 输出:
//   d_offsets[i] = i * nSamples。
//
// 作用:
//   告诉 CUB 哪些元素属于同一个 segment，也就是“每个特征自己的那一行响应”。
__global__ void FillSegmentOffsetsKernel(int* __restrict__ d_offsets,
                                         int featureCount,
                                         int nSamples) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i > featureCount) {
        return;
    }
    // 每个特征一整段，offset 形如 [0, N, 2N, 3N, ...]，
    // CUB 就知道要把 [feature][sample] 看成 featureCount 段独立数组。
    d_offsets[i] = i * nSamples;
}

// 输入:
//   d_best: 当前 tile 内每个特征已经求好的最优切分结果。
//   featureBegin: 当前 tile 在全局特征池里的起始下标。
//   featureCount: 当前 tile 的特征数量。
//   d_usedMask: 已被当前 cascade stage 选中过的特征掩码，可为空。
//
// 输出:
//   d_out[0] 写入这个 tile 中整体最优的 TileBestCandidate。
//
// 核心思路:
//   每个线程遍历若干个特征，先找自己的局部最优，再用 shared memory 做块内归约。
__global__ void SelectBestInTileKernel(const FeatureBestSplit* __restrict__ d_best,
                                       int featureBegin,
                                       int featureCount,
                                       const uint8_t* __restrict__ d_usedMask,
                                       TileBestCandidate* __restrict__ d_out) {
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ TileBestLocal sBest[kThresholdThreads];

    // 每个线程先在自己负责的 feature 子集里找局部最优，再做块内归约。
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
//
// 这也是训练积分图要转置存储的根本原因：
// 如果仍按 [sample][iiPixel] 布局，warp 内线程会跨大步长读内存，带宽利用率很差。
//
// 输入:
//   d_features: 全局特征池。
//   featureBegin / featureCount: 当前 tile 的特征范围。
//   d_sumT: 转置积分图 [iiArea][nSamples]。
//   d_invNorm: 每个样本的归一化系数，可为空。
//   nSamples: 样本数。
//
// 输出:
//   d_respOut: [featureCount][nSamples]，每个元素都是一个“特征-样本”响应值。
//
// 并行映射:
//   一个 block 对应一个特征；
//   block 内线程沿 sample 维并行，一个线程负责一个样本。
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

    // 一个 block 只处理一个特征，所以把 HaarFeature 放进 shared memory，
    // 避免 256 个线程都重复从 global memory 读取同一份结构体。
    __shared__ HaarFeature f;
    if (threadIdx.x == 0) {
        f = d_features[featureBegin + featureLocal];
    }
    __syncthreads();

    // 每次读积分图时，所有线程访问同一个角点偏移 p，但 sample 不同，
    // 这正好匹配转置布局的连续内存模式。
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

    // 训练时使用方差归一化，减轻亮度整体偏移对响应值的影响。
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
//
// 为什么一定要“先排序再扫阈值”：
// 对决策 stump 来说，最优阈值只可能出现在相邻不同响应值之间。
// 先排序后，就能在线性时间扫描所有候选切分点，而不是 O(N^2) 暴力尝试。
//
// 输入:
//   d_resp: [featureCount][nSamples]，每个特征在所有样本上的响应值。
//   d_sortedIdx: [featureCount][nSamples]，每个特征对应的有序样本下标。
//   d_label: 每个样本的真实标签，+1 表示正样本，-1 表示负样本。
//   d_active: 可选的 active mask，某些负样本在后续 stage 中可能已经失活。
//   d_weight: AdaBoost 当前轮样本权重。
//
// 输出:
//   d_outBest: 当前 tile 内每个特征各自最优的 FeatureBestSplit。
//
// 并行映射:
//   一个 block 对应一个特征；
//   block 内线程共同扫描这个特征的全部候选切分点。
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

    // 这里用 CUB 做块级归约/扫描，减少自己手写 shared-memory 前缀和的复杂度。
    using PairReduce = cub::BlockReduce<WeightPair, kThresholdThreads>;
    using FloatScan = cub::BlockScan<float, kThresholdThreads>;
    using BestReduce = cub::BlockReduce<LocalBest, kThresholdThreads>;
    __shared__ typename PairReduce::TempStorage sPairReduce;
    __shared__ typename FloatScan::TempStorage sScan;
    __shared__ typename BestReduce::TempStorage sBestReduce;

    __shared__ float sTotalPos;
    __shared__ float sTotalNeg;
    __shared__ float sCarryPos;
    __shared__ float sCarryNeg;

    // --------------------
    // Step A: global totals
    // --------------------
    // 先统计整条排序序列上的正/负总权重，后面算“左边错多少 + 右边错多少”要用。
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

    const WeightPair totals = PairReduce(sPairReduce).Sum(WeightPair{localPos, localNeg});
    __syncthreads();

    if (tid == 0) {
        sTotalPos = totals.pos;
        sTotalNeg = totals.neg;
        sCarryPos = 0.0f;
        sCarryNeg = 0.0f;
    }
    __syncthreads();

    // 每个线程先记录自己的最优切分候选，最后再做块内最小归约。
    LocalBest best{FLT_MAX, -1, +1};

    // ---------------------------------------------------------
    // Step B: tile-by-tile prefix scan over sorted sample order
    // ---------------------------------------------------------
    // blockDim.x 个线程一次处理一小段有序样本，并维护“左边累计权重”。
    // 这样既能并行，又能在整个有序序列上完成线性扫描。
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

        float prefPos = 0.0f;
        float prefNeg = 0.0f;
        float blockAggPos = 0.0f;
        float blockAggNeg = 0.0f;
        // 正负权重分开扫描，因为后面两种 parity 的错误率都要分别用到。
        FloatScan(sScan).InclusiveSum(wp, prefPos, blockAggPos);
        __syncthreads();
        FloatScan(sScan).InclusiveSum(wn, prefNeg, blockAggNeg);
        __syncthreads();

        const float carryPos = sCarryPos;
        const float carryNeg = sCarryNeg;

        // Evaluate split at k (between k and k+1).
        // 也就是把前 k+1 个样本放左边，剩下样本放右边。
        if (k < nSamples - 1) {
            const int si0 = static_cast<int>(ord[k]);
            const int si1 = static_cast<int>(ord[k + 1]);
            const float v0 = respRow[si0];
            const float v1 = respRow[si1];

            // 只有当相邻响应值真的不同，阈值放在它们中间才有意义。
            if (v0 + kValueEps < v1) {
                const float wPosLeft = carryPos + prefPos;
                const float wNegLeft = carryNeg + prefNeg;
                const float wPosRight = sTotalPos - wPosLeft;
                const float wNegRight = sTotalNeg - wNegLeft;

                // parity = -1:
                // 左边预测 +1，右边预测 -1
                // 错误 = 左边负样本权重 + 右边正样本权重
                const float errP = wNegLeft + wPosRight;
                LocalBest candP{errP, k, -1};
                best = betterBest(best, candP);

                // parity = +1:
                // 左边预测 -1，右边预测 +1
                // 错误 = 左边正样本权重 + 右边负样本权重
                const float errN = wPosLeft + wNegRight;
                LocalBest candN{errN, k, +1};
                best = betterBest(best, candN);
            }
        }

        // Update carries (cumulative left-side sums) once per tile.
        if (tid == 0) {
            sCarryPos += blockAggPos;
            sCarryNeg += blockAggNeg;
        }
        __syncthreads();
    }

    // -------------------------------------
    // Step C: block min reduction for best.
    // -------------------------------------
    // 到这里每个线程已经看过一部分 split，把最优结果再归约成“这个特征的全局最优切分”。
    const LocalBest blockBest = BestReduce(sBestReduce).Reduce(best, BetterBestOp{});
    __syncthreads();

    if (tid == 0) {
        FeatureBestSplit out{};
        out.bestErr = blockBest.err;
        out.bestSplitPos = blockBest.split;
        out.bestParity = static_cast<int8_t>(blockBest.parity);

        if (out.bestSplitPos >= 0 && out.bestSplitPos < nSamples - 1) {
            const int si0 = static_cast<int>(ord[out.bestSplitPos]);
            const int si1 = static_cast<int>(ord[out.bestSplitPos + 1]);
            // theta 取相邻两个响应的中点，保证阈值落在“可分开”的区间内部。
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
    // 统一把 CUDA API 返回码映射到项目自己的 Status 枚举，便于上层统一处理错误。
    return err == cudaSuccess ? Status::kOk : Status::kCudaError;
}

AdaBoostTrainer::AdaBoostTrainer(const WindowSpec& win, int maxSamples, int maxFeaturesPerTile)
    : win_(win), maxSamples_(maxSamples), maxFeaturesPerTile_(maxFeaturesPerTile) {
    // 这个类把训练热点阶段需要的工作缓冲区一次性申请好，
    // 后续每轮 boosting 直接复用，避免频繁 cudaMalloc/cudaFree。
    if (maxSamples_ <= 0 || maxFeaturesPerTile_ <= 0) {
        throw std::invalid_argument("AdaBoostTrainer: invalid maxSamples/maxFeaturesPerTile");
    }

    // 下面这些缓冲区几乎都与 tile 大小线性相关，
    // 所以 feature tile 太大时会显著增加显存占用和排序工作集。
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
        // dResponseBuffer_:
        // 保存当前 tile 内每个特征在全部样本上的响应值，是后续排序和阈值扫描的输入。
        alloc_or_throw(reinterpret_cast<void**>(&dResponseBuffer_), respBytes,
                       "AdaBoostTrainer: cudaMalloc dResponseBuffer_ failed");
        // dBestBuffer_:
        // 保存当前 tile 内每个特征独立求出的最优切分。
        alloc_or_throw(reinterpret_cast<void**>(&dBestBuffer_), bestBytes,
                       "AdaBoostTrainer: cudaMalloc dBestBuffer_ failed");
        // dTileBestBuffer_:
        // 保存当前 tile 最终挑出的全局最优候选。
        alloc_or_throw(reinterpret_cast<void**>(&dTileBestBuffer_), tileBestBytes,
                       "AdaBoostTrainer: cudaMalloc dTileBestBuffer_ failed");

        // 下面四个缓冲区全部是 segmented sort 的工作集。
        alloc_or_throw(reinterpret_cast<void**>(&dSortKeysOut_), respBytes,
                       "AdaBoostTrainer: cudaMalloc dSortKeysOut_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dSortValsIn_), idxBytes,
                       "AdaBoostTrainer: cudaMalloc dSortValsIn_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dSortValsOut_), idxBytes,
                       "AdaBoostTrainer: cudaMalloc dSortValsOut_ failed");
        alloc_or_throw(reinterpret_cast<void**>(&dSegmentOffsets_), segOfsBytes,
                       "AdaBoostTrainer: cudaMalloc dSegmentOffsets_ failed");

        // 先让 CUB“试运行”一次，只查询临时缓冲区大小，不真正排序。
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
        // 构造失败时要手动回收已经申请成功的显存，保证异常安全。
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
    // RAII 析构：谁申请谁释放，避免训练代码到处写清理逻辑。
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
    // 复用 move 赋值逻辑，避免两处写重复代码。
    *this = std::move(other);
}

AdaBoostTrainer& AdaBoostTrainer::operator=(AdaBoostTrainer&& other) noexcept {
    // 输入:
    //   other: 被移动的源对象。
    //
    // 输出:
    //   当前对象接管 other 的全部 device 资源；other 变成空壳。
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

    // move 语义的目的不是“复制数据”，而是把显存所有权安全地转交给新对象。
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

    // 被移动走的对象要清空成可析构状态，避免 double free。
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
    // 这里只“绑定视图”而不复制训练数据，避免每一轮都重复搬大块显存。
    // 输入:
    //   iiSet: 来自 FaceVisionEngine 的积分图视图，布局固定为 [iiArea][N]。
    //   labels: 样本标签和 active mask。
    //   weights: 样本权重。
    //
    // 输出:
    //   内部成员 iiSet_ / labels_ / weights_ 被更新；
    //   若样本数变化，还会重建排序布局相关的索引和 offsets。
    if (!iiSet.d_sumT || !iiSet.d_invNorm || !labels.d_label || !weights.d_weight) {
        return Status::kInvalidArg;
    }
    if (iiSet.nSamples <= 0 || iiSet.nSamples > maxSamples_) {
        return Status::kInvalidArg;
    }
    // 排序 value 缓冲区把 sample index 压成 uint16_t，以节省显存和排序带宽；
    // 所以样本数不能超过 65535。
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
        // 只有样本数变化时才重建排序布局，这样每轮 boosting 都能复用初始化结果。
        const int nSamples = iiSet.nSamples;
        {
            // 为每个特征段初始化 [0,1,2,...,nSamples-1] 的样本编号。
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
            // 建立 segmented sort 所需的段偏移表。
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
    // 输出布局是 [featureCount][nSamples]，便于后面直接做“每个特征一段”的分段排序。
    // 输入:
    //   dFeatures: 全局 device 特征池。
    //   featureBegin: 当前 tile 起始位置。
    //   featureCount: 当前 tile 的特征数量。
    //   dOutResp: 可选输出缓冲区。
    //
    // 输出:
    //   每个特征对每个样本的归一化响应值。
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
    // 输入必须已经是“响应值 + 每个特征自己的排序索引”。
    // 输入:
    //   tile: 当前 tile 的视图，里面包含特征范围、响应矩阵、排序索引。
    //
    // 输出:
    //   每个特征最优的 threshold / parity / error。
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
    // 输入:
    //   dResp: [featureCount][nSamples]。
    //   featureCount: 当前 tile 的特征数。
    //   dOutSortedIdx: 可选输出地址，若为空则结果保存在内部缓冲区。
    //
    // 输出:
    //   每个特征一行对应一个有序样本下标数组。
    //
    // 性能关键点:
    //   这里排序的是“响应值 + 样本编号”对，而不是把样本数据本身搬来搬去。
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

    // 一次 CUB 分段排序同时排完整个 tile，减少 kernel launch 次数。
    // 这比“每个特征单独排一次”更适合 GPU。
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

    // 默认结果就放在成员缓冲区里；如果调用方想拿一份额外副本，再做一次 D2D 拷贝。
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
    // dUsedFeatureMask 用来防止同一 stage 里重复选择同一个 Haar 特征。
    // 输入:
    //   featureBegin / featureCount: 当前 tile 在全局特征池中的范围。
    //   dUsedFeatureMask: 已经被当前 stage 选过的特征掩码，可为空。
    //
    // 输出:
    //   dOut 中写入一个 tile 级最佳候选。
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
