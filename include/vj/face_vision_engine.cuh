#pragma once

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "common_types.cuh"

namespace vj {

// FaceVisionEngine 是项目里的“视觉底座”：
// 1) 生成和上传 Haar 特征池；
// 2) 计算积分图/平方积分图；
// 3) 把训练后的模型重排成检测 kernel 友好的格式；
// 4) 执行多尺度滑窗检测，并在 CPU 上做结果分组。
//
// 训练阶段会用它提供积分图和特征池；
// 检测阶段会用它跑完整推理流程。
class FaceVisionEngine {
public:
    // Basic / Core / All 控制特征池规模：
    // Basic: 基础 upright 特征；
    // Core: 增加更丰富的 upright 模板；
    // All: 进一步包含 tilted 特征。
    enum class FeatureMode : int { Basic = 0, Core = 1, All = 2 };

    // 输入:
    //   win: 基础检测窗口大小。
    //   maxSamples: 最多同时处理多少张样本图。
    //   maxImageW / maxImageH: 内部缓冲区支持的最大图像尺寸。
    //   enableTilted: 是否允许生成 tilted 特征。
    //   maxDetections: 检测输出缓冲区上限。
    //
    // 输出:
    //   构造函数会一次性申请积分图、金字塔、模型和检测结果相关缓冲区。
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

    // 统计某种模式下总共有多少个 Haar 特征。
    static int countHaarFeatures(const WindowSpec& win, FeatureMode mode);
    // 枚举指定模式下的所有 Haar 特征，并以 host vector 形式返回。
    static std::vector<HaarFeature> buildHaarFeaturePool(const WindowSpec& win, FeatureMode mode);

    // 输入:
    //   hFeatures: host 侧特征池。
    //
    // 输出:
    //   在 device 端保存一份特征池，供训练或检测 kernel 直接访问。
    Status uploadFeaturePool(const std::vector<HaarFeature>& hFeatures,
                             cudaStream_t stream = nullptr);

    const HaarFeature* deviceFeaturePool() const noexcept { return dFeaturePool_; }
    int featurePoolSize() const noexcept { return featurePoolSize_; }

    // Core A1: 计算一批图像的积分图和平方积分图，并转置成 [iiArea][N] 布局。
    //
    // 输入:
    //   dGrayBatch: 灰度图批次，原始布局按 [sample][row][col] 排列。
    //   imageW / imageH / imageStride: 单张图的尺寸与步长。
    //   nSamples: 批次大小。
    //   computeInvNorm: 是否顺带计算每个样本的方差归一化因子。
    //
    // 输出:
    //   内部 dSumT_ / dSqsumT_ / dInvNorm_ 被更新。
    //
    // 优化点:
    //   先行扫描，再列扫描并转置，兼顾前缀和计算效率和后续训练访存效率。
    Status computeIntegralBatchTransposed(const uint8_t* dGrayBatch,
                                          int imageW,
                                          int imageH,
                                          int imageStride,
                                          int nSamples,
                                          bool computeInvNorm = true,
                                          cudaStream_t stream = nullptr);

    // 返回最近一次 computeIntegralBatchTransposed 结果的轻量视图。
    IntegralImageSetT integralSetView() const noexcept;

    // 输入:
    //   dUsedFeatures: 训练后真正被模型使用到的特征子集。
    //   usedFeatureCount: 该子集大小。
    //   dStumps: GPU 检测侧紧凑 stump 数组。
    //   stumpCount: stump 总数。
    //   hStages: host 侧 stage 描述。
    //
    // 输出:
    //   内部模型缓冲区和 LUT 被更新，后续 detectMultiScale 可以直接使用。
    //
    // 优化点:
    //   会把特征坐标预解码成 dx/dy LUT，减少检测 kernel 内的整数解码开销。
    Status setCascadeModel(const HaarFeature* dUsedFeatures,
                           int usedFeatureCount,
                           const GpuStump4* dStumps,
                           int stumpCount,
                           const std::vector<CascadeStage>& hStages,
                           cudaStream_t stream = nullptr);

    // 释放当前上传的 cascade 模型及其 LUT，但保留基础积分图/金字塔缓冲区。
    Status clearCascadeModel(cudaStream_t stream = nullptr);

    // Core A3: 多尺度检测主流程。
    //
    // 输入:
    //   dImageGray: 待检测灰度图。
    //   imageW / imageH / imageStride: 图像尺寸。
    //   scaleFactor: 图像金字塔缩放因子。
    //   minNeighbors: 最终分组时的最小邻居数。
    //   minObjectSize: 最小检测目标尺寸。
    //   maxDetections: 本次调用允许输出的最大候选框数。
    //   applyGrouping: 是否对原始候选框做 groupRectangles。
    //
    // 输出:
    //   outDetections: 最终检测框列表。
    //
    // 关键优化:
    //   图像金字塔 + 级联 early rejection + 大尺度跳步扫描。
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

    // 积分图构建过程中的中间结果与最终结果。
    int32_t* dRowPrefix_ = nullptr;      // [N][H][W+1]
    int32_t* dRowPrefixSq_ = nullptr;    // [N][H][W+1]
    int32_t* dSumT_ = nullptr;           // [iiArea][N]
    int32_t* dSqsumT_ = nullptr;         // [iiArea][N]
    float* dInvNorm_ = nullptr;          // [N]

    // 图像金字塔的双缓冲，交替用于不同尺度层。
    uint8_t* dPyrBufA_ = nullptr;
    uint8_t* dPyrBufB_ = nullptr;

    // 当前上传的 cascade 模型及其 LUT。
    HaarFeature* dModelFeatures_ = nullptr;
    int32_t* dLutDx_[kLutRects][kLutCorners] = {};
    int32_t* dLutDy_[kLutRects][kLutCorners] = {};
    float* dLutW_[kLutRects] = {};
    uint8_t* dLutRectCount_ = nullptr;
    int modelFeatureCount_ = 0;
    int modelFeatureCapacity_ = 0;
    int modelLutCapacity_ = 0;
    GpuStump4* dModelStumps_ = nullptr;
    int modelStumpCount_ = 0;
    int modelStumpCapacity_ = 0;
    CascadeStage* dModelStages_ = nullptr;
    int modelStageCount_ = 0;
    int modelStageCapacity_ = 0;

    // 检测输出缓冲区：device 端累计候选框，host 端用 pinned memory 回收结果。
    Detection* dDetectOut_ = nullptr;
    int* dDetectCount_ = nullptr;
    Detection* hDetectPinned_ = nullptr;
    int* hDetectCountPinned_ = nullptr;

    // CPU 版 groupRectangles，尽量模拟 OpenCV 的后处理行为。
    std::vector<Detection> groupRectanglesCpu(const std::vector<Detection>& in,
                                              int minNeighbors,
                                              float eps) const;
};

} // namespace vj
