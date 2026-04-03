#include "vj/model_io.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>

namespace vj {
namespace ModelIO {
namespace {

// 模型文件头:
// 1) magic 用来快速判断“这是不是我们自己定义的模型文件”；
// 2) version 便于将来扩展格式时做兼容；
// 3) 后面的计数字段让读取端可以一次性知道需要读多少条特征 / stump / stage。
struct ModelHeader {
    char magic[8];
    uint32_t version;
    int32_t winW;
    int32_t winH;
    uint64_t featureCount;
    uint64_t stumpCount;
    uint64_t stageCount;
};

// 固定 magic，避免把任意二进制文件误当成模型读取。
constexpr char kMagic[8] = {'V', 'J', 'C', 'B', 'I', 'N', '1', '\0'};
// 当前模型格式版本。以后如果结构变化，可以升级版本号并在 load 时分支处理。
constexpr uint32_t kVersion = 1;

template <typename T>
bool writeArray(FILE* fp, const std::vector<T>& v) {
    // 输入:
    //   fp: 已打开的二进制文件句柄。
    //   v: 要整块写入的数组。
    //
    // 输出:
    //   返回是否完整写入成功。
    // 空数组直接视为成功，避免 fwrite(nullptr, 0, ...) 这种边界情况让逻辑变复杂。
    if (v.empty()) {
        return true;
    }
    // 这里直接整块写入，前提是 T 是可平铺写盘的 POD/标准布局类型。
    return std::fwrite(v.data(), sizeof(T), v.size(), fp) == v.size();
}

template <typename T>
bool readArray(FILE* fp, std::vector<T>& v, uint64_t n) {
    // 输入:
    //   fp: 已打开的二进制文件句柄。
    //   n: 需要读取多少个元素。
    //
    // 输出:
    //   v: 被 resize 并填充为读取结果。
    //   返回值表示是否完整读取成功。
    if (n == 0) {
        v.clear();
        return true;
    }
    // 先 resize 再整块读取，避免一条条 push_back 带来的额外分配和循环开销。
    v.resize(static_cast<size_t>(n));
    return std::fread(v.data(), sizeof(T), v.size(), fp) == v.size();
}

} // namespace

bool saveCascadeModel(const char* filepath,
                      const WindowSpec& win,
                      const std::vector<HaarFeature>& features,
                      const std::vector<GpuStump4>& stumps,
                      const std::vector<CascadeStage>& stages) {
    // 输入:
    //   filepath: 输出模型路径。
    //   win: 检测窗口大小。
    //   features: 模型中会用到的 Haar 特征集合。
    //   stumps: GPU 检测侧使用的紧凑 stump 数组。
    //   stages: 级联 stage 描述。
    //
    // 输出:
    //   返回模型是否成功写入磁盘。
    //
    // 文件格式:
    //   [ModelHeader][HaarFeature...][GpuStump4...][CascadeStage...]
    // 训练出的模型如果这几项缺任何一项，检测阶段都没法正确工作，所以直接拒绝保存。
    if (!filepath || !filepath[0] || win.winW <= 0 || win.winH <= 0 || features.empty() || stumps.empty() || stages.empty()) {
        return false;
    }

    // 用 C 风格 FILE 是因为这里做的是简单顺序二进制 IO，代码更短，也更接近“按字节写盘”的语义。
    FILE* fp = std::fopen(filepath, "wb");
    if (!fp) {
        return false;
    }

    ModelHeader h{};
    // 先写文件头，告诉加载端这个文件里接下来会出现什么。
    std::memcpy(h.magic, kMagic, sizeof(kMagic));
    h.version = kVersion;
    h.winW = win.winW;
    h.winH = win.winH;
    h.featureCount = static_cast<uint64_t>(features.size());
    h.stumpCount = static_cast<uint64_t>(stumps.size());
    h.stageCount = static_cast<uint64_t>(stages.size());

    // 文件布局非常直接:
    // [ModelHeader][HaarFeature 数组][GpuStump4 数组][CascadeStage 数组]
    // 这样做的优点是实现简单、读取快、序列化成本低。
    bool ok = std::fwrite(&h, sizeof(h), 1, fp) == 1 &&
              writeArray(fp, features) &&
              writeArray(fp, stumps) &&
              writeArray(fp, stages);

    // 无论写成功失败都要 fclose，避免文件句柄泄漏。
    std::fclose(fp);
    return ok;
}

bool loadCascadeModel(const char* filepath,
                      WindowSpec& win,
                      std::vector<HaarFeature>& features,
                      std::vector<GpuStump4>& stumps,
                      std::vector<CascadeStage>& stages) {
    // 输入:
    //   filepath: 待读取模型路径。
    //
    // 输出:
    //   win / features / stumps / stages 被填充为模型内容。
    //   返回值表示模型是否读取并通过基本一致性校验。
    // 路径为空说明调用方参数就错了，不必继续尝试。
    if (!filepath || !filepath[0]) {
        return false;
    }

    FILE* fp = std::fopen(filepath, "rb");
    if (!fp) {
        return false;
    }

    ModelHeader h{};
    // 第一步一定先读头，因为后面数组的长度都靠头里的计数字段决定。
    if (std::fread(&h, sizeof(h), 1, fp) != 1) {
        std::fclose(fp);
        return false;
    }

    // 这里做“快速一致性校验”：
    // 1) magic/version 不对，说明文件格式不匹配；
    // 2) 窗口尺寸或各数组个数非法，说明文件损坏或内容不完整。
    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0 || h.version != kVersion ||
        h.winW <= 0 || h.winH <= 0 || h.featureCount == 0 || h.stumpCount == 0 || h.stageCount == 0) {
        std::fclose(fp);
        return false;
    }

    // 按 header 记录的数量整块恢复各数组。
    if (!readArray(fp, features, h.featureCount) ||
        !readArray(fp, stumps, h.stumpCount) ||
        !readArray(fp, stages, h.stageCount)) {
        std::fclose(fp);
        return false;
    }

    // WindowSpec 不单独存在于数组里，所以最后从头部回填给调用方。
    win.winW = h.winW;
    win.winH = h.winH;
    std::fclose(fp);
    return true;
}

} // namespace ModelIO
} // namespace vj
