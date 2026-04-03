#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "common_types.cuh"

namespace vj {

// AdaBoostTrainer 专门负责训练阶段里最贵的一段：
// 给定一批训练样本、样本权重和 Haar 特征池，
// 在 GPU 上高效找出“当前这一轮 AdaBoost 应该选哪个最优弱分类器(stump)”。
//
// 它不负责完整的 stage 训练循环，也不直接保存模型；
// 它提供的是几个可复用的底层算子：
// 1) 计算特征响应；
// 2) 按特征对样本响应排序；
// 3) 扫描阈值求每个特征的最优切分；
// 4) 从一个 feature tile 里选出整体最优候选。
class AdaBoostTrainer {
public:
    // 输入:
    //   win: 检测窗口大小，主要用于和整个训练流程保持语义一致。
    //   maxSamples: 这个 trainer 允许处理的最大样本数。
    //   maxFeaturesPerTile: 单次批处理的最大特征数，也就是 feature tile 大小。
    //
    // 输出:
    //   构造完成后，会一次性申请好训练热点所需的显存缓冲区。
    //
    // 说明:
    //   这里把 tile 大小作为构造参数，是因为很多内部缓冲区大小都正比于
    //   maxSamples * maxFeaturesPerTile，提前固定后便于复用显存。
    AdaBoostTrainer(const WindowSpec& win, int maxSamples, int maxFeaturesPerTile);
    ~AdaBoostTrainer();

    AdaBoostTrainer(const AdaBoostTrainer&) = delete;
    AdaBoostTrainer& operator=(const AdaBoostTrainer&) = delete;
    AdaBoostTrainer(AdaBoostTrainer&& other) noexcept;
    AdaBoostTrainer& operator=(AdaBoostTrainer&& other) noexcept;

    // 输入:
    //   iiSet: 转置布局的积分图 / 平方积分图 / invNorm 视图。
    //   labels: 样本标签和可选的 active mask。
    //   weights: AdaBoost 当前轮的样本权重。
    //
    // 输出:
    //   不复制真实样本数据，只把这些 device 指针和元信息绑定到 trainer 内部。
    //
    // 说明:
    //   这是一个“绑定视图”的轻量操作，正常会在样本集准备好后调用一次，
    //   后续多轮弱分类器搜索会复用这些绑定结果。
    Status bindTrainingSet(const IntegralImageSetT& iiSet,
                           const LabelsView& labels,
                           const WeightsView& weights);

    // Core-1: 在转置积分图 [iiArea][N] 上计算一段 Haar 特征的响应值。
    //
    // 输入:
    //   dFeatures: device 端 Haar 特征池基地址。
    //   featureBegin: 本次 tile 在全局特征池中的起始下标。
    //   featureCount: 本次需要计算多少个特征。
    //   dOutResp: 可选输出缓冲区，布局为 [featureCount][nSamples]。
    //             为空时默认写入内部 dResponseBuffer_。
    //
    // 输出:
    //   每个特征对每个样本的归一化响应值。
    //
    // 优化点:
    //   训练积分图采用 [iiPixel][sample] 转置布局，使 warp 在访问固定角点偏移时
    //   可以对 sample 维做合并访存。
    Status evaluateFeatureResponses(const HaarFeature* dFeatures,
                                    int featureBegin,
                                    int featureCount,
                                    float* dOutResp = nullptr,
                                    cudaStream_t stream = nullptr) const;

    // Core-2: 对一个 feature tile 内的每个特征做阈值扫描，求最优弱分类切分。
    //
    // 输入:
    //   tile.d_resp: [featureCount][nSamples] 的响应矩阵。
    //   tile.d_sortedIdx: 每个特征对应一段已按响应值排好序的样本下标。
    //
    // 输出:
    //   对每个特征输出一个 FeatureBestSplit，记录最优误差、阈值、切分点和 parity。
    //
    // 说明:
    //   这是 stump 训练里最核心的一步。因为响应已排序，所以每个特征只需 O(N)
    //   扫一遍所有候选切分点，而不是 O(N^2) 暴力尝试。
    Status evaluateAndFindThreshold(const FeatureTileView& tile,
                                    FeatureBestSplit* dOutBestPerFeature = nullptr,
                                    cudaStream_t stream = nullptr) const;

    // 从当前 tile 内所有特征的最优切分中，再选出一个 tile 级最优候选。
    //
    // 输入:
    //   featureBegin / featureCount: 当前 tile 覆盖的全局特征范围。
    //   dUsedFeatureMask: 可选 mask，标记哪些特征已经在当前 stage 被选中过。
    //
    // 输出:
    //   一个 TileBestCandidate，表示这个 tile 中当前最值得加入 AdaBoost 的特征。
    Status selectBestInTile(int featureBegin,
                            int featureCount,
                            const uint8_t* dUsedFeatureMask,
                            TileBestCandidate* dOut = nullptr,
                            cudaStream_t stream = nullptr) const;

    // 分段排序: 对 tile 内每个特征自己的响应行独立排序。
    //
    // 输入:
    //   dResp: [featureCount][nSamples]，每行是一个特征在全部样本上的响应。
    //
    // 输出:
    //   dOutSortedIdx: [featureCount][nSamples]，每行保存“按响应从小到大排序后的样本下标”。
    //   若 dOutSortedIdx 为空，则结果默认留在内部 dSortValsOut_。
    //
    // 优化点:
    //   使用 CUB DeviceSegmentedRadixSort 一次排序整块 tile，
    //   避免对每个特征单独发起一次排序调用。
    Status sortSamplesPerFeature(const float* dResp,
                                 int featureCount,
                                 uint16_t* dOutSortedIdx = nullptr,
                                 cudaStream_t stream = nullptr);

    float* responseBuffer() noexcept { return dResponseBuffer_; }
    const float* responseBuffer() const noexcept { return dResponseBuffer_; }

    FeatureBestSplit* bestBuffer() noexcept { return dBestBuffer_; }
    const FeatureBestSplit* bestBuffer() const noexcept { return dBestBuffer_; }
    TileBestCandidate* tileBestBuffer() noexcept { return dTileBestBuffer_; }
    const TileBestCandidate* tileBestBuffer() const noexcept { return dTileBestBuffer_; }
    const uint16_t* sortedIndexBuffer() const noexcept { return dSortValsOut_; }

    int maxSamples() const noexcept { return maxSamples_; }
    int maxFeaturesPerTile() const noexcept { return maxFeaturesPerTile_; }

private:
    static Status fromCuda(cudaError_t err);

    WindowSpec win_{};
    int maxSamples_ = 0;
    int maxFeaturesPerTile_ = 0;

    IntegralImageSetT iiSet_{};
    LabelsView labels_{};
    WeightsView weights_{};

    // RAII-owned device buffers for this boosting core
    float* dResponseBuffer_ = nullptr;            // [maxFeaturesPerTile][maxSamples]
    FeatureBestSplit* dBestBuffer_ = nullptr;     // [maxFeaturesPerTile]
    TileBestCandidate* dTileBestBuffer_ = nullptr; // [1]

    // Segmented sort working set (allocated once, reused per boosting round)
    float* dSortKeysOut_ = nullptr;               // [maxFeaturesPerTile][maxSamples]
    uint16_t* dSortValsIn_ = nullptr;             // [maxFeaturesPerTile][maxSamples]
    uint16_t* dSortValsOut_ = nullptr;            // [maxFeaturesPerTile][maxSamples]
    int* dSegmentOffsets_ = nullptr;              // [maxFeaturesPerTile + 1]
    void* dSortTempStorage_ = nullptr;            // CUB temp buffer
    size_t sortTempStorageBytes_ = 0;
    int sortLayoutSamples_ = 0;                   // nSamples used for dSortValsIn_/dSegmentOffsets_
    bool sortLayoutReady_ = false;
};

} // namespace vj
