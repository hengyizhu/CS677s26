#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace vj {

// 项目内部统一的状态码。
// 之所以不用直接把 cudaError_t 暴露到所有地方，是为了把 CUDA 错误、
// 参数错误和运行时逻辑错误统一到一个更简单的接口层。
enum class Status : int {
    kOk = 0,
    kInvalidArg = 1,
    kCudaError = 2,
    kOutOfMemory = 3,
    kRuntimeError = 4
};

struct WindowSpec {
    // Viola-Jones 的基础检测窗口大小。
    // 训练和检测都默认围绕这个固定窗口建立特征与级联模型。
    int winW;
    int winH;
};

struct FastRect {
    // 这是积分图版矩形表示：
    // p0/p1/p2/p3 是四个角在积分图中的一维下标，
    // w 是这个矩形在 Haar 特征线性组合中的权重。
    //
    // 好处是检测和训练时都不需要再显式存 x/y/w/h，
    // 直接 4 次积分图访问就能得到矩形和。
    int p0, p1, p2, p3;
    float w;
};

struct HaarFeature {
    // 一个 Haar 特征最多由 3 个矩形组成。
    // r0/r1 必用，r2 只在三矩形模板时有效。
    FastRect r0;
    FastRect r1;
    FastRect r2;
    uint8_t rectCount;   // 2 or 3
    uint8_t tilted;      // 0/1
    uint16_t reserved;
};

struct IntegralImageSetT {
    // 这一组字段描述“当前训练批次的积分图视图”。
    // 核心布局是 [iiArea][N]，也就是固定积分图像素位置时，样本维连续。
    // 这是训练 kernel 能高效合并访存的关键。
    int nSamples;
    int iiWidth;
    int iiHeight;
    int iiArea;
    const int32_t* d_sumT;      // [iiArea][N]
    const int32_t* d_sqsumT;    // [iiArea][N]
    const int32_t* d_tiltedT;   // optional
    const float* d_invNorm;     // [N]
};

struct LabelsView {
    // 标签和 active mask 被单独抽成 view，
    // 是为了让训练核心直接绑定 device 指针而不是复制 host 数据。
    int nSamples;
    const int8_t* d_label;      // +1 / -1
    const uint8_t* d_active;    // 1 active, 0 inactive (nullable)
};

struct WeightsView {
    // AdaBoost 当前轮样本权重视图。
    int nSamples;
    float* d_weight;            // [N]
};

struct FeatureTileView {
    // 训练时不会一次处理全部特征，而是按 tile 分块。
    // 这个结构就是“当前特征块”的轻量描述。
    int featureBegin;
    int featureCount;
    const HaarFeature* d_features; // base pool pointer
    const float* d_resp;           // [featureCount][N]
    const uint16_t* d_sortedIdx;   // [featureCount][N]
};

struct FeatureBestSplit {
    // 针对“单个特征”做完阈值扫描之后的最优结果。
    float bestErr;
    float bestTheta;
    int32_t bestSplitPos;
    int8_t bestParity;            // +1 / -1
    int8_t pad0, pad1, pad2;
};

struct TileBestCandidate {
    // 针对“一个 feature tile”做完归约后的最优候选。
    float bestErr;
    float bestTheta;
    int32_t featureIdx;
    int8_t bestParity;
    int8_t pad0, pad1, pad2;
};

struct WeakClassifier {
    // host 侧更易读的弱分类器表示。
    // 训练过程中先以这个结构累计，再转成检测侧更紧凑的 GpuStump4。
    int featureIdx;
    float theta;
    int8_t parity;
    int8_t pad0, pad1, pad2;
    float err;
    float alpha;
    float leftVal;
    float rightVal;
};

struct CascadeStage {
    // 级联的一个 stage 对应若干个弱分类器的线性组合。
    // first / ntrees 指向一段连续的 weak/stump。
    int first;
    int ntrees;
    float threshold;
};


struct Detection {
    // 检测输出框。
    // scaleIdx 保留来自哪一层图像金字塔，score 保留最后一个 stage 的得分。
    int x;
    int y;
    int w;
    int h;
    int scaleIdx;
    float score;
};

struct alignas(16) GpuStump4 {
    // GPU 检测侧的紧凑 stump 表示。
    // 之所以压成 float4:
    // 1) 16 字节对齐，读取简单；
    // 2) 检测 kernel 一次 load 就能取到一个 stump 的全部核心参数。
    float4 st; // {featureIdx(as_float_bits), theta, leftVal, rightVal}
};

} // namespace vj
