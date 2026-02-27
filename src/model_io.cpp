#include "vj/model_io.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>

namespace vj {
namespace ModelIO {
namespace {

struct ModelHeader {
    char magic[8];
    uint32_t version;
    int32_t winW;
    int32_t winH;
    uint64_t featureCount;
    uint64_t stumpCount;
    uint64_t stageCount;
};

constexpr char kMagic[8] = {'V', 'J', 'C', 'B', 'I', 'N', '1', '\0'};
constexpr uint32_t kVersion = 1;

template <typename T>
bool writeArray(FILE* fp, const std::vector<T>& v) {
    if (v.empty()) {
        return true;
    }
    return std::fwrite(v.data(), sizeof(T), v.size(), fp) == v.size();
}

template <typename T>
bool readArray(FILE* fp, std::vector<T>& v, uint64_t n) {
    if (n == 0) {
        v.clear();
        return true;
    }
    v.resize(static_cast<size_t>(n));
    return std::fread(v.data(), sizeof(T), v.size(), fp) == v.size();
}

} // namespace

bool saveCascadeModel(const char* filepath,
                      const WindowSpec& win,
                      const std::vector<HaarFeature>& features,
                      const std::vector<GpuStump4>& stumps,
                      const std::vector<CascadeStage>& stages) {
    if (!filepath || !filepath[0] || win.winW <= 0 || win.winH <= 0 || features.empty() || stumps.empty() || stages.empty()) {
        return false;
    }

    FILE* fp = std::fopen(filepath, "wb");
    if (!fp) {
        return false;
    }

    ModelHeader h{};
    std::memcpy(h.magic, kMagic, sizeof(kMagic));
    h.version = kVersion;
    h.winW = win.winW;
    h.winH = win.winH;
    h.featureCount = static_cast<uint64_t>(features.size());
    h.stumpCount = static_cast<uint64_t>(stumps.size());
    h.stageCount = static_cast<uint64_t>(stages.size());

    bool ok = std::fwrite(&h, sizeof(h), 1, fp) == 1 &&
              writeArray(fp, features) &&
              writeArray(fp, stumps) &&
              writeArray(fp, stages);

    std::fclose(fp);
    return ok;
}

bool loadCascadeModel(const char* filepath,
                      WindowSpec& win,
                      std::vector<HaarFeature>& features,
                      std::vector<GpuStump4>& stumps,
                      std::vector<CascadeStage>& stages) {
    if (!filepath || !filepath[0]) {
        return false;
    }

    FILE* fp = std::fopen(filepath, "rb");
    if (!fp) {
        return false;
    }

    ModelHeader h{};
    if (std::fread(&h, sizeof(h), 1, fp) != 1) {
        std::fclose(fp);
        return false;
    }

    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0 || h.version != kVersion ||
        h.winW <= 0 || h.winH <= 0 || h.featureCount == 0 || h.stumpCount == 0 || h.stageCount == 0) {
        std::fclose(fp);
        return false;
    }

    if (!readArray(fp, features, h.featureCount) ||
        !readArray(fp, stumps, h.stumpCount) ||
        !readArray(fp, stages, h.stageCount)) {
        std::fclose(fp);
        return false;
    }

    win.winW = h.winW;
    win.winH = h.winH;
    std::fclose(fp);
    return true;
}

} // namespace ModelIO
} // namespace vj

