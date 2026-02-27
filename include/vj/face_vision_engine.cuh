#pragma once

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "common_types.cuh"

namespace vj {

class FaceVisionEngine {
public:
    enum class FeatureMode : int { Basic = 0, Core = 1, All = 2 };

    FaceVisionEngine(const WindowSpec& win,
                     int maxSamples,
                     int maxImageW,
                     int maxImageH,
                     bool enableTilted = true,
                     int maxDetections = 200000);
    ~FaceVisionEngine();

    FaceVisionEngine(const FaceVisionEngine&) = delete;
    FaceVisionEngine& operator=(const FaceVisionEngine&) = delete;
    FaceVisionEngine(FaceVisionEngine&& other) noexcept;
    FaceVisionEngine& operator=(FaceVisionEngine&& other) noexcept;

    static int countHaarFeatures(const WindowSpec& win, FeatureMode mode);
    static std::vector<HaarFeature> buildHaarFeaturePool(const WindowSpec& win, FeatureMode mode);

    Status uploadFeaturePool(const std::vector<HaarFeature>& hFeatures,
                             cudaStream_t stream = nullptr);

    const HaarFeature* deviceFeaturePool() const noexcept { return dFeaturePool_; }
    int featurePoolSize() const noexcept { return featurePoolSize_; }

    // Core A1: 2D prefix scan + transpose write layout [iiArea][N]
    Status computeIntegralBatchTransposed(const uint8_t* dGrayBatch,
                                          int imageW,
                                          int imageH,
                                          int imageStride,
                                          int nSamples,
                                          cudaStream_t stream = nullptr);

    IntegralImageSetT integralSetView() const noexcept;

    Status setCascadeModel(const HaarFeature* dUsedFeatures,
                           int usedFeatureCount,
                           const GpuStump4* dStumps,
                           int stumpCount,
                           const std::vector<CascadeStage>& hStages,
                           cudaStream_t stream = nullptr);

    Status clearCascadeModel(cudaStream_t stream = nullptr);

    // Core A3: pyramid + cascade early rejection + CPU NMS
    Status detectMultiScale(const uint8_t* dImageGray,
                            int imageW,
                            int imageH,
                            int imageStride,
                            float scaleFactor,
                            int minNeighbors,
                            int minObjectSize,
                            int maxDetections,
                            std::vector<Detection>& outDetections,
                            bool applyGrouping = true,
                            cudaStream_t stream = nullptr);

private:
    static constexpr int kLutRects = 3;
    static constexpr int kLutCorners = 4;

    static Status fromCuda(cudaError_t err);

    WindowSpec win_{};
    int maxSamples_ = 0;
    int maxImageW_ = 0;
    int maxImageH_ = 0;
    bool enableTilted_ = true;
    int maxDetections_ = 0;

    int featurePoolSize_ = 0;
    HaarFeature* dFeaturePool_ = nullptr;

    int currentSamples_ = 0;
    int currentImageW_ = 0;
    int currentImageH_ = 0;

    int32_t* dRowPrefix_ = nullptr;      // [N][H][W+1]
    int32_t* dRowPrefixSq_ = nullptr;    // [N][H][W+1]
    int32_t* dSumT_ = nullptr;           // [iiArea][N]
    int32_t* dSqsumT_ = nullptr;         // [iiArea][N]
    float* dInvNorm_ = nullptr;          // [N]

    uint8_t* dPyrBufA_ = nullptr;
    uint8_t* dPyrBufB_ = nullptr;

    HaarFeature* dModelFeatures_ = nullptr;
    int32_t* dLutDx_[kLutRects][kLutCorners] = {};
    int32_t* dLutDy_[kLutRects][kLutCorners] = {};
    float* dLutW_[kLutRects] = {};
    uint8_t* dLutRectCount_ = nullptr;
    int modelFeatureCount_ = 0;
    GpuStump4* dModelStumps_ = nullptr;
    int modelStumpCount_ = 0;
    CascadeStage* dModelStages_ = nullptr;
    int modelStageCount_ = 0;

    Detection* dDetectOut_ = nullptr;
    int* dDetectCount_ = nullptr;
    Detection* hDetectPinned_ = nullptr;
    int* hDetectCountPinned_ = nullptr;

    std::vector<Detection> groupRectanglesCpu(const std::vector<Detection>& in,
                                              int minNeighbors,
                                              float eps) const;
};

} // namespace vj
