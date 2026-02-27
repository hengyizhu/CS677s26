#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace vj {

enum class Status : int {
    kOk = 0,
    kInvalidArg = 1,
    kCudaError = 2,
    kOutOfMemory = 3,
    kRuntimeError = 4
};

struct WindowSpec {
    int winW;
    int winH;
};

struct FastRect {
    int p0, p1, p2, p3;
    float w;
};

struct HaarFeature {
    FastRect r0;
    FastRect r1;
    FastRect r2;
    uint8_t rectCount;   // 2 or 3
    uint8_t tilted;      // 0/1
    uint16_t reserved;
};

struct IntegralImageSetT {
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
    int nSamples;
    const int8_t* d_label;      // +1 / -1
    const uint8_t* d_active;    // 1 active, 0 inactive (nullable)
};

struct WeightsView {
    int nSamples;
    float* d_weight;            // [N]
};

struct FeatureTileView {
    int featureBegin;
    int featureCount;
    const HaarFeature* d_features; // base pool pointer
    const float* d_resp;           // [featureCount][N]
    const uint16_t* d_sortedIdx;   // [featureCount][N]
};

struct FeatureBestSplit {
    float bestErr;
    float bestTheta;
    int32_t bestSplitPos;
    int8_t bestParity;            // +1 / -1
    int8_t pad0, pad1, pad2;
};

struct WeakClassifier {
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
    int first;
    int ntrees;
    float threshold;
};


struct Detection {
    int x;
    int y;
    int w;
    int h;
    int scaleIdx;
    float score;
};

struct alignas(16) GpuStump4 {
    float4 st; // {featureIdx(as_float_bits), theta, leftVal, rightVal}
};

} // namespace vj
