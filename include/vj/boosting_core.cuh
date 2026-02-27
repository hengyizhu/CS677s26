#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "common_types.cuh"

namespace vj {

class AdaBoostTrainer {
public:
    AdaBoostTrainer(const WindowSpec& win, int maxSamples, int maxFeaturesPerTile);
    ~AdaBoostTrainer();

    AdaBoostTrainer(const AdaBoostTrainer&) = delete;
    AdaBoostTrainer& operator=(const AdaBoostTrainer&) = delete;
    AdaBoostTrainer(AdaBoostTrainer&& other) noexcept;
    AdaBoostTrainer& operator=(AdaBoostTrainer&& other) noexcept;

    Status bindTrainingSet(const IntegralImageSetT& iiSet,
                           const LabelsView& labels,
                           const WeightsView& weights);

    // Core-1: evaluate feature responses on transposed integral images [iiArea][N]
    Status evaluateFeatureResponses(const HaarFeature* dFeatures,
                                    int featureBegin,
                                    int featureCount,
                                    float* dOutResp = nullptr,
                                    cudaStream_t stream = nullptr) const;

    // Core-2: threshold sweep for each feature (1 block = 1 feature)
    Status evaluateAndFindThreshold(const FeatureTileView& tile,
                                    FeatureBestSplit* dOutBestPerFeature = nullptr,
                                    cudaStream_t stream = nullptr) const;

    // Segmented sort: for each feature row, sort sample indices by response value.
    // Input dResp layout: [featureCount][nSamples]
    // Output dOutSortedIdx layout: [featureCount][nSamples]
    Status sortSamplesPerFeature(const float* dResp,
                                 int featureCount,
                                 uint16_t* dOutSortedIdx = nullptr,
                                 cudaStream_t stream = nullptr);

    float* responseBuffer() noexcept { return dResponseBuffer_; }
    const float* responseBuffer() const noexcept { return dResponseBuffer_; }

    FeatureBestSplit* bestBuffer() noexcept { return dBestBuffer_; }
    const FeatureBestSplit* bestBuffer() const noexcept { return dBestBuffer_; }
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

    // Segmented sort working set (allocated once, reused per boosting round)
    float* dSortKeysIn_ = nullptr;                // [maxFeaturesPerTile][maxSamples]
    float* dSortKeysOut_ = nullptr;               // [maxFeaturesPerTile][maxSamples]
    uint16_t* dSortValsIn_ = nullptr;             // [maxFeaturesPerTile][maxSamples]
    uint16_t* dSortValsOut_ = nullptr;            // [maxFeaturesPerTile][maxSamples]
    int* dSegmentOffsets_ = nullptr;              // [maxFeaturesPerTile + 1]
    void* dSortTempStorage_ = nullptr;            // CUB temp buffer
    size_t sortTempStorageBytes_ = 0;
};

} // namespace vj
