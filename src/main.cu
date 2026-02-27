#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "vj/boosting_core.cuh"
#include "vj/face_vision_engine.cuh"
#include "vj/model_io.hpp"

namespace {

using vj::AdaBoostTrainer;
using vj::CascadeStage;
using vj::Detection;
using vj::FaceVisionEngine;
using vj::FeatureTileView;
using vj::GpuStump4;
using vj::HaarFeature;
using vj::IntegralImageSetT;
using vj::LabelsView;
using vj::Status;
using vj::TileBestCandidate;
using vj::WeakClassifier;
using vj::WeightsView;
using vj::WindowSpec;

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;
    explicit CudaBuffer(size_t count) { allocate(count); }
    ~CudaBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }
    CudaBuffer& operator=(CudaBuffer&& o) noexcept {
        if (this == &o) return *this;
        if (ptr_) cudaFree(ptr_);
        ptr_ = o.ptr_;
        count_ = o.count_;
        o.ptr_ = nullptr;
        o.count_ = 0;
        return *this;
    }

    bool allocate(size_t count) {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
        if (count == 0) return true;
        if (cudaMalloc(&ptr_, count * sizeof(T)) != cudaSuccess) return false;
        count_ = count;
        return true;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

struct GrayImage {
    int w = 0;
    int h = 0;
    std::vector<uint8_t> data;
};

std::string statusToString(Status s) {
    switch (s) {
        case Status::kOk: return "kOk";
        case Status::kInvalidArg: return "kInvalidArg";
        case Status::kCudaError: return "kCudaError";
        case Status::kOutOfMemory: return "kOutOfMemory";
        case Status::kRuntimeError: return "kRuntimeError";
        default: return "Unknown";
    }
}

bool loadRawSampleBin(const std::string& path,
                      int samplePixels,
                      std::vector<uint8_t>& out,
                      int& outCount) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "[ERR] Cannot open: " << path << "\n";
        return false;
    }

    ifs.seekg(0, std::ios::end);
    const std::streamoff sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (sz <= 0) {
        std::cerr << "[ERR] Empty file: " << path << "\n";
        return false;
    }
    if (sz % samplePixels != 0) {
        std::cerr << "[ERR] File size is not divisible by samplePixels: " << path << "\n";
        return false;
    }

    out.resize(static_cast<size_t>(sz));
    if (!ifs.read(reinterpret_cast<char*>(out.data()), sz)) {
        std::cerr << "[ERR] Failed to read: " << path << "\n";
        return false;
    }
    outCount = static_cast<int>(sz / samplePixels);
    return true;
}

std::string shellQuote(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

bool regenerateStageNegativesFromCache(const std::string& scriptPath,
                                       const std::string& cacheDir,
                                       const std::string& imagesDir,
                                       const std::string& outDir,
                                       int stageIdx,
                                       int numNeg,
                                       int winSize,
                                       int seed,
                                       std::string& outBinPath) {
    std::ostringstream binName;
    binName << "non_faces_stage";
    if (stageIdx < 10) {
        binName << "0";
    }
    binName << stageIdx << ".bin";
    outBinPath = outDir + "/" + binName.str();

    std::ostringstream cmd;
    cmd << "python3 " << shellQuote(scriptPath)
        << " --sample-stage"
        << " --reuse-existing"
        << " --stage-idx " << stageIdx
        << " --num-neg " << numNeg
        << " --win " << winSize
        << " --seed " << seed
        << " --cache-dir " << shellQuote(cacheDir)
        << " --images-dir " << shellQuote(imagesDir)
        << " --out-dir " << shellQuote(outDir);

    const int rc = std::system(cmd.str().c_str());
    return rc == 0;
}

std::vector<std::string> collectPgmPaths(const std::string& rootDir) {
    std::vector<std::string> out;
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(rootDir, ec)) return out;
    for (fs::recursive_directory_iterator it(rootDir, ec), end; it != end; it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file()) continue;
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".pgm") {
            out.push_back(it->path().string());
        }
    }
    return out;
}

void cropResizeTo24(const GrayImage& img, const Detection& d, std::vector<uint8_t>& out24) {
    const int outW = 24;
    const int outH = 24;
    out24.resize(static_cast<size_t>(outW) * outH);

    const float sx = static_cast<float>(std::max(1, d.w));
    const float sy = static_cast<float>(std::max(1, d.h));

    for (int oy = 0; oy < outH; ++oy) {
        for (int ox = 0; ox < outW; ++ox) {
            const float srcFx = static_cast<float>(d.x) + (static_cast<float>(ox) + 0.5f) * (sx / outW) - 0.5f;
            const float srcFy = static_cast<float>(d.y) + (static_cast<float>(oy) + 0.5f) * (sy / outH) - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(srcFx)), 0, img.w - 1);
            const int y0 = std::clamp(static_cast<int>(std::floor(srcFy)), 0, img.h - 1);
            const int x1 = std::min(x0 + 1, img.w - 1);
            const int y1 = std::min(y0 + 1, img.h - 1);
            const float ax = srcFx - static_cast<float>(x0);
            const float ay = srcFy - static_cast<float>(y0);
            const float p00 = static_cast<float>(img.data[static_cast<size_t>(y0) * img.w + x0]);
            const float p01 = static_cast<float>(img.data[static_cast<size_t>(y0) * img.w + x1]);
            const float p10 = static_cast<float>(img.data[static_cast<size_t>(y1) * img.w + x0]);
            const float p11 = static_cast<float>(img.data[static_cast<size_t>(y1) * img.w + x1]);
            const float top = p00 + (p01 - p00) * ax;
            const float bot = p10 + (p11 - p10) * ax;
            const float val = top + (bot - top) * ay;
            out24[static_cast<size_t>(oy) * outW + ox] =
                static_cast<uint8_t>(std::clamp(static_cast<int>(val + 0.5f), 0, 255));
        }
    }
}

bool nextToken(std::istream& is, std::string& tok) {
    tok.clear();
    while (is.good()) {
        char c = static_cast<char>(is.peek());
        if (c == '#') {
            std::string line;
            std::getline(is, line);
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            is.get();
            continue;
        }
        break;
    }
    if (!is.good()) return false;
    is >> tok;
    return !tok.empty();
}

bool loadPGM(const std::string& path, GrayImage& img) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "[ERR] Cannot open PGM: " << path << "\n";
        return false;
    }

    std::string tok;
    if (!nextToken(ifs, tok) || tok != "P5") {
        std::cerr << "[ERR] Unsupported PGM format (need P5): " << path << "\n";
        return false;
    }
    if (!nextToken(ifs, tok)) return false;
    img.w = std::stoi(tok);
    if (!nextToken(ifs, tok)) return false;
    img.h = std::stoi(tok);
    if (!nextToken(ifs, tok)) return false;
    const int maxVal = std::stoi(tok);
    if (img.w <= 0 || img.h <= 0 || maxVal <= 0 || maxVal > 255) {
        std::cerr << "[ERR] Invalid PGM metadata\n";
        return false;
    }

    // Skip one whitespace before raw bytes
    ifs.get();

    img.data.resize(static_cast<size_t>(img.w) * img.h);
    if (!ifs.read(reinterpret_cast<char*>(img.data.data()), static_cast<std::streamsize>(img.data.size()))) {
        std::cerr << "[ERR] Failed to read PGM pixel payload\n";
        return false;
    }
    return true;
}

bool savePPM(const std::string& path, int w, int h, const std::vector<uint8_t>& rgb) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[ERR] Cannot write PPM: " << path << "\n";
        return false;
    }
    ofs << "P6\n" << w << " " << h << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(ofs);
}

void drawRect(std::vector<uint8_t>& rgb, int w, int h, const Detection& d) {
    const int x0 = std::max(0, d.x);
    const int y0 = std::max(0, d.y);
    const int x1 = std::min(w - 1, d.x + d.w - 1);
    const int y1 = std::min(h - 1, d.y + d.h - 1);
    if (x0 > x1 || y0 > y1) return;

    auto setPix = [&](int x, int y) {
        const size_t idx = (static_cast<size_t>(y) * w + x) * 3;
        rgb[idx + 0] = 255;
        rgb[idx + 1] = 255;
        rgb[idx + 2] = 255;
    };

    for (int t = 0; t < 2; ++t) {
        const int yt = std::min(h - 1, y0 + t);
        const int yb = std::max(0, y1 - t);
        for (int x = x0; x <= x1; ++x) {
            setPix(x, yt);
            setPix(x, yb);
        }

        const int xl = std::min(w - 1, x0 + t);
        const int xr = std::max(0, x1 - t);
        for (int y = y0; y <= y1; ++y) {
            setPix(xl, y);
            setPix(xr, y);
        }
    }
}

__global__ void UpdateWeightsDiscreteKernel(const float* __restrict__ resp,
                                            const int8_t* __restrict__ labels,
                                            const uint8_t* __restrict__ active,
                                            float* __restrict__ weights,
                                            int n,
                                            float theta,
                                            int parity,
                                            float coeffC) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    if (active && active[i] == 0) {
        return;
    }

    const float r = resp[i];
    const int pred = (parity > 0)
                         ? ((r < theta) ? -1 : +1)
                         : ((r < theta) ? +1 : -1);

    if (pred != static_cast<int>(labels[i])) {
        weights[i] *= expf(coeffC);
    }
}

__global__ void NormalizeWeightsKernel(float* __restrict__ weights,
                                       int n,
                                       const float* __restrict__ sumW) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const float s = sumW[0];
    if (s > 1e-30f) {
        weights[i] /= s;
    }
}

__global__ void ApplyActiveMaskToWeightsKernel(float* __restrict__ weights,
                                               const uint8_t* __restrict__ active,
                                               int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    if (active && active[i] == 0) {
        weights[i] = 0.0f;
    }
}

__global__ void AccumulateStrongScoreKernel(const float* __restrict__ resp,
                                            float* __restrict__ strong,
                                            int n,
                                            float theta,
                                            float leftVal,
                                            float rightVal) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    strong[i] += (resp[i] < theta) ? leftVal : rightVal;
}

__global__ void FillArrayKernel(float* __restrict__ arr, int n, float v) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) arr[i] = v;
}

__device__ __forceinline__ TileBestCandidate betterTileCandidate(const TileBestCandidate& a,
                                                                 const TileBestCandidate& b) {
    if (b.bestErr < a.bestErr) return b;
    if (b.bestErr > a.bestErr) return a;
    if (b.featureIdx < a.featureIdx) return b;
    return a;
}

__global__ void ReduceTileBestCandidatesKernel(const TileBestCandidate* __restrict__ in,
                                               int n,
                                               TileBestCandidate* __restrict__ out) {
    __shared__ TileBestCandidate s[256];
    const int tid = static_cast<int>(threadIdx.x);
    TileBestCandidate local{};
    local.bestErr = INFINITY;
    local.bestTheta = 0.0f;
    local.featureIdx = INT_MAX;
    local.bestParity = +1;
    local.pad0 = local.pad1 = local.pad2 = 0;

    for (int i = tid; i < n; i += blockDim.x) {
        local = betterTileCandidate(local, in[i]);
    }
    s[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s[tid] = betterTileCandidate(s[tid], s[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = s[0];
    }
}

}  // namespace

int runTrain(int argc, char** argv) {
    constexpr int kPaperStages = 38;
    constexpr int kPaperTotalFeatures = 6061;
    constexpr int kPaperPosPerStage = 10000;
    constexpr int kPaperNegPerStage = 10000;
    constexpr float kPaperScaleFactor = 1.25f;
    constexpr float kPaperMinHitRate = 0.99f;
    constexpr float kPaperMaxFalseAlarm = 0.40f;
    const int modelArgIdx = argc - 1;
    if (modelArgIdx < 1) {
        std::cerr << "Usage: cuda_hello train [faces_u8.bin] [numNeg]"
                  << " [maxWeakPerStage] [maxDetections] [minNeighbors] [scaleFactor] [maxStages]"
                  << " [minHitRate] [maxFalseAlarm] [minObjectSize]"
                  << " [stageNegCacheDir] [stageNegImagesDir] [stageNegSeed] [hardNegCandidateMultiplier]"
                  << " <out_model.bin>\n";
        return 1;
    }
    const std::string modelOutPath = argv[modelArgIdx];
    const std::string posPath = (modelArgIdx > 1) ? argv[1] : "faces_u8.bin";
    int numNeg = kPaperNegPerStage;

    const WindowSpec win{24, 24};
    const int samplePixels = win.winW * win.winH;
    int maxWeakPerStage = 250;
    int maxStages = kPaperStages;
    int maxDetections = 200000;
    int minNeighbors = 4;
    int minObjectSize = 24;
    float scaleFactor = kPaperScaleFactor;
    int hardNegCandidateMultiplier = 3;
    std::string stageNegCacheDir;
    std::string stageNegImagesDir = "data_cache/coco_val2017_nonperson_pgm";
    int stageNegSeed = 677;

    if (modelArgIdx > 2) {
        try {
            numNeg = std::max(1, std::stoi(argv[2]));
        } catch (...) {
            std::cerr << "[ERR] Invalid numNeg argument: " << argv[2] << "\n";
            return 1;
        }
    }

    if (modelArgIdx > 3) {
        try {
            maxWeakPerStage = std::max(1, std::stoi(argv[3]));
        } catch (...) {
            std::cerr << "[ERR] Invalid maxWeakPerStage argument: " << argv[3] << "\n";
            return 1;
        }
    }
    if (modelArgIdx > 4) {
        try {
            maxDetections = std::max(100, std::stoi(argv[4]));
        } catch (...) {
            std::cerr << "[ERR] Invalid maxDetections argument: " << argv[4] << "\n";
            return 1;
        }
    }
    if (modelArgIdx > 5) {
        try {
            minNeighbors = std::max(1, std::stoi(argv[5]));
        } catch (...) {
            std::cerr << "[ERR] Invalid minNeighbors argument: " << argv[5] << "\n";
            return 1;
        }
    }
    if (modelArgIdx > 6) {
        try {
            scaleFactor = std::stof(argv[6]);
            if (scaleFactor <= 1.0f) {
                std::cerr << "[ERR] scaleFactor must be > 1.0\n";
                return 1;
            }
        } catch (...) {
            std::cerr << "[ERR] Invalid scaleFactor argument: " << argv[6] << "\n";
            return 1;
        }
    }
    if (modelArgIdx > 7) {
        try {
            maxStages = std::max(1, std::stoi(argv[7]));
        } catch (...) {
            std::cerr << "[ERR] Invalid maxStages argument: " << argv[7] << "\n";
            return 1;
        }
    }
    float minHitRate = kPaperMinHitRate;
    float maxFalseAlarm = kPaperMaxFalseAlarm;
    if (modelArgIdx > 8) {
        try {
            minHitRate = std::stof(argv[8]);
            if (minHitRate <= 0.0f || minHitRate > 1.0f) {
                std::cerr << "[ERR] minHitRate must be in (0, 1]\n";
                return 1;
            }
        } catch (...) {
            std::cerr << "[ERR] Invalid minHitRate argument: " << argv[8] << "\n";
            return 1;
        }
    }
    if (modelArgIdx > 9) {
        try {
            maxFalseAlarm = std::stof(argv[9]);
            if (maxFalseAlarm < 0.0f || maxFalseAlarm > 1.0f) {
                std::cerr << "[ERR] maxFalseAlarm must be in [0, 1]\n";
                return 1;
            }
        } catch (...) {
            std::cerr << "[ERR] Invalid maxFalseAlarm argument: " << argv[9] << "\n";
            return 1;
        }
    }
    if (modelArgIdx > 10) {
        try {
            minObjectSize = std::max(1, std::stoi(argv[10]));
        } catch (...) {
            std::cerr << "[ERR] Invalid minObjectSize argument: " << argv[10] << "\n";
            return 1;
        }
    }
    if (modelArgIdx > 11) {
        stageNegCacheDir = argv[11];
    }
    if (modelArgIdx > 12) {
        stageNegImagesDir = argv[12];
    }
    if (modelArgIdx > 13) {
        try {
            stageNegSeed = std::stoi(argv[13]);
        } catch (...) {
            std::cerr << "[ERR] Invalid stageNegSeed argument: " << argv[13] << "\n";
            return 1;
        }
    }
    if (modelArgIdx > 14) {
        try {
            hardNegCandidateMultiplier = std::max(1, std::stoi(argv[14]));
        } catch (...) {
            std::cerr << "[ERR] Invalid hardNegCandidateMultiplier argument: " << argv[14] << "\n";
            return 1;
        }
    }

    if (numNeg != kPaperNegPerStage) {
        std::cout << "[INFO] numNeg is fixed to paper-style per-stage setting: " << kPaperNegPerStage << "\n";
        numNeg = kPaperNegPerStage;
    }
    if (maxStages > kPaperStages) {
        std::cout << "[INFO] maxStages capped to " << kPaperStages << "\n";
        maxStages = kPaperStages;
    }

    const std::string stageNegScriptPath = "scripts/stage_negatives.py";
    const std::string stageNegOutDir = stageNegCacheDir + "/stage_negatives";

    std::vector<uint8_t> posRaw;
    int numPosPool = 0;

    std::cout << "[INFO] Config: maxWeakPerStage=" << maxWeakPerStage
              << " maxStages=" << maxStages
              << " scaleFactor=" << scaleFactor
              << " minNeighbors=" << minNeighbors
              << " minObjectSize=" << minObjectSize
              << " maxDetections=" << maxDetections
              << " minHitRate=" << minHitRate
              << " maxFalseAlarm=" << maxFalseAlarm
              << " numNeg=" << numNeg
              << " hardNegCandMul=" << hardNegCandidateMultiplier << "\n";

    if (stageNegCacheDir.empty()) {
        std::cerr << "[ERR] stageNegCacheDir is required. Dynamic negatives are mandatory.\n";
        std::cerr << "Usage: cuda_hello train ... <out_model.bin>\n";
        return 1;
    }

    if (!loadRawSampleBin(posPath, samplePixels, posRaw, numPosPool)) {
        return 1;
    }
    if (numPosPool <= 0) {
        std::cerr << "[ERR] no positive samples in " << posPath << "\n";
        return 1;
    }

    const int numPos = std::min(numPosPool, kPaperPosPerStage);
    if (numPosPool > kPaperPosPerStage) {
        std::cout << "[INFO] positive pool=" << numPosPool
                  << ", per-stage positives fixed to " << numPos << " (randomly sampled each stage)\n";
    } else {
        std::cout << "[INFO] positive pool=" << numPosPool
                  << ", using all positives per stage\n";
    }

    const int nSamples = numPos + numNeg;
    if (nSamples <= 0) {
        std::cerr << "[ERR] No training samples\n";
        return 1;
    }

    std::vector<uint8_t> trainRaw(static_cast<size_t>(nSamples) * samplePixels, 0);
    std::vector<uint8_t> selectedPosRaw(static_cast<size_t>(numPos) * samplePixels, 0);
    auto fillStagePositives = [&](int stageSeed) {
        if (numPosPool <= numPos) {
            std::memcpy(selectedPosRaw.data(), posRaw.data(), selectedPosRaw.size());
            return;
        }
        std::vector<int> idx(static_cast<size_t>(numPosPool));
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(static_cast<uint32_t>(stageSeed ^ 0x7f4a7c15));
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int i = 0; i < numPos; ++i) {
            const int src = idx[static_cast<size_t>(i)];
            std::memcpy(selectedPosRaw.data() + static_cast<size_t>(i) * samplePixels,
                        posRaw.data() + static_cast<size_t>(src) * samplePixels,
                        samplePixels);
        }
    };
    fillStagePositives(stageNegSeed);
    std::memcpy(trainRaw.data(), selectedPosRaw.data(), selectedPosRaw.size());

    std::vector<int8_t> hLabels(nSamples, -1);
    for (int i = 0; i < numPos; ++i) hLabels[i] = +1;
    std::vector<uint8_t> hActive(nSamples, 1);

    const float wPos = (numPos > 0) ? (0.5f / static_cast<float>(numPos)) : 0.0f;
    const float wNeg = (numNeg > 0) ? (0.5f / static_cast<float>(numNeg)) : 0.0f;
    std::vector<float> hWeights(nSamples, 0.0f);
    for (int i = 0; i < numPos; ++i) hWeights[i] = wPos;
    for (int i = numPos; i < nSamples; ++i) hWeights[i] = wNeg;

    const int maxImageW = win.winW;
    const int maxImageH = win.winH;
    const int miningMaxImageW = std::max(2048, maxImageW);
    const int miningMaxImageH = std::max(2048, maxImageH);

    // Tile size controls memory footprint and kernel launch count.
    const int featureTile = 1024;

    // Split training and inference engines to avoid allocating training buffers
    // at demo-image resolution for all samples (huge VRAM waste).
    FaceVisionEngine trainEngine(win, nSamples, win.winW, win.winH, false, 1);
    FaceVisionEngine detectEngine(win, 1, miningMaxImageW, miningMaxImageH, false, maxDetections);
    AdaBoostTrainer trainer(win, nSamples, featureTile);

    std::cout << "[INFO] Building Haar feature pool...\n";
    std::vector<HaarFeature> hFeatures =
        FaceVisionEngine::buildHaarFeaturePool(win, FaceVisionEngine::FeatureMode::Core);
    const int numFeatures = static_cast<int>(hFeatures.size());
    const int numFeatureTiles = (numFeatures + featureTile - 1) / featureTile;
    std::cout << "[INFO] Feature count: " << numFeatures << "\n";

    Status st = trainEngine.uploadFeaturePool(hFeatures);
    if (st != Status::kOk) {
        std::cerr << "[ERR] uploadFeaturePool failed: " << statusToString(st) << "\n";
        return 1;
    }

    CudaBuffer<uint8_t> dTrain(trainRaw.size());
    CudaBuffer<int8_t> dLabel(static_cast<size_t>(nSamples));
    CudaBuffer<uint8_t> dActive(static_cast<size_t>(nSamples));
    CudaBuffer<float> dWeight(static_cast<size_t>(nSamples));
    CudaBuffer<uint8_t> dMiningImage(static_cast<size_t>(miningMaxImageW) * miningMaxImageH);

    if (cudaMemcpy(dTrain.data(), trainRaw.data(), trainRaw.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dLabel.data(), hLabels.data(), static_cast<size_t>(nSamples) * sizeof(int8_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dActive.data(), hActive.data(), static_cast<size_t>(nSamples) * sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dWeight.data(), hWeights.data(), static_cast<size_t>(nSamples) * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "[ERR] cudaMemcpy to device failed\n";
        return 1;
    }

    st = trainEngine.computeIntegralBatchTransposed(dTrain.data(), win.winW, win.winH, win.winW, nSamples);
    if (st != Status::kOk) {
        std::cerr << "[ERR] computeIntegralBatchTransposed failed: " << statusToString(st) << "\n";
        return 1;
    }

    IntegralImageSetT iiSet = trainEngine.integralSetView();
    LabelsView labelsView{nSamples, dLabel.data(), dActive.data()};
    WeightsView weightsView{nSamples, dWeight.data()};

    st = trainer.bindTrainingSet(iiSet, labelsView, weightsView);
    if (st != Status::kOk) {
        std::cerr << "[ERR] bindTrainingSet failed: " << statusToString(st) << "\n";
        return 1;
    }

    CudaBuffer<float> dWeightSum(1);
    size_t reduceBytes = 0;
    cub::DeviceReduce::Sum(nullptr, reduceBytes, dWeight.data(), dWeightSum.data(), nSamples);
    CudaBuffer<uint8_t> dReduceTemp(reduceBytes);

    std::vector<WeakClassifier> selectedWeaks;
    selectedWeaks.reserve(static_cast<size_t>(maxStages) * maxWeakPerStage);
    std::vector<CascadeStage> stages;
    stages.reserve(static_cast<size_t>(maxStages));
    TileBestCandidate hTileBest{};
    CudaBuffer<TileBestCandidate> dTileBestPerRound(static_cast<size_t>(numFeatureTiles));
    CudaBuffer<TileBestCandidate> dRoundBest(1);
    CudaBuffer<float> dStrong(static_cast<size_t>(nSamples));
    CudaBuffer<uint8_t> dUsedFeatureMask(static_cast<size_t>(numFeatures));
    std::vector<float> hStrong(static_cast<size_t>(nSamples), 0.0f);
    std::vector<float> posScores(static_cast<size_t>(numPos), 0.0f);
    std::vector<float> hWeightsDbg;
    std::vector<float> hRespDbg;

    int debugEvery = 0;
    if (const char* dbgEnv = std::getenv("VJ_DEBUG_EVERY")) {
        try {
            debugEvery = std::max(0, std::stoi(dbgEnv));
        } catch (...) {
            debugEvery = 0;
        }
    }
    if (debugEvery > 0) {
        hWeightsDbg.assign(static_cast<size_t>(nSamples), 0.0f);
        hRespDbg.assign(static_cast<size_t>(nSamples), 0.0f);
        std::cout << "[INFO] Debug logging enabled: VJ_DEBUG_EVERY=" << debugEvery << "\n";
    }

    const dim3 blk(256, 1, 1);
    const dim3 grd(static_cast<unsigned>((nSamples + 255) / 256), 1, 1);

    auto normalizeWeights = [&]() -> bool {
        ApplyActiveMaskToWeightsKernel<<<grd, blk>>>(dWeight.data(), dActive.data(), nSamples);
        if (cudaGetLastError() != cudaSuccess) {
            return false;
        }
        if (cub::DeviceReduce::Sum(dReduceTemp.data(),
                                   reduceBytes,
                                   dWeight.data(),
                                   dWeightSum.data(),
                                   nSamples) != cudaSuccess) {
            return false;
        }
        NormalizeWeightsKernel<<<grd, blk>>>(dWeight.data(), nSamples, dWeightSum.data());
        return cudaGetLastError() == cudaSuccess;
    };

    auto resetStageWeights = [&](int activeNegCount) -> bool {
        if (activeNegCount <= 0) {
            return false;
        }
        std::fill(hWeights.begin(), hWeights.end(), 0.0f);
        const float wPos = 0.5f / static_cast<float>(numPos);
        const float wNeg = 0.5f / static_cast<float>(activeNegCount);
        for (int i = 0; i < numPos; ++i) {
            hWeights[i] = wPos;
            hActive[i] = 1;
        }
        for (int i = numPos; i < nSamples; ++i) {
            hWeights[i] = (hActive[i] != 0) ? wNeg : 0.0f;
        }
        return cudaMemcpy(dWeight.data(), hWeights.data(),
                          static_cast<size_t>(nSamples) * sizeof(float),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    };

    auto calibrateStageThreshold = [&](const std::vector<float>& strongScores,
                                       float& outThreshold,
                                       float& outHitRate) {
        for (int i = 0; i < numPos; ++i) {
            posScores[static_cast<size_t>(i)] = strongScores[static_cast<size_t>(i)];
        }
        const int minPass = std::max(1, static_cast<int>(std::ceil(minHitRate * static_cast<float>(numPos))));
        const int cutIdx = numPos - minPass;
        auto cutIt = posScores.begin() + cutIdx;
        std::nth_element(posScores.begin(), cutIt, posScores.end());
        const float cutVal = *cutIt;

        int passGe = 0;
        int passGt = 0;
        for (int i = 0; i < numPos; ++i) {
            const float s = strongScores[static_cast<size_t>(i)];
            if (s > cutVal) {
                ++passGt;
                ++passGe;
            } else if (s == cutVal) {
                ++passGe;
            }
        }

        const float threshold = (passGt >= minPass)
                                    ? std::nextafter(cutVal, std::numeric_limits<float>::infinity())
                                    : cutVal;
        const int posPass = (passGt >= minPass) ? passGt : passGe;
        outThreshold = threshold;
        outHitRate = static_cast<float>(posPass) / static_cast<float>(numPos);
    };

    cudaDeviceSynchronize();
    const auto tTrain0 = std::chrono::high_resolution_clock::now();
    std::cout << "[TRAIN] Start cascade training: maxStages=" << maxStages
              << " maxWeakPerStage=" << maxWeakPerStage
              << " totalFeatureBudget=" << kPaperTotalFeatures << "\n";

    for (int stageIdx = 0; stageIdx < maxStages; ++stageIdx) {
        if (static_cast<int>(selectedWeaks.size()) >= kPaperTotalFeatures) {
            std::cout << "[TRAIN] Reached total feature budget " << kPaperTotalFeatures
                      << ", stop at stage " << stageIdx << "\n";
            break;
        }
        const int perStageSeed = stageNegSeed + stageIdx * 1000003;
        fillStagePositives(perStageSeed);
        std::vector<uint8_t> selectedNegRaw(static_cast<size_t>(numNeg) * samplePixels, 0);
        int hardSelected = 0;
        std::vector<uint8_t> stageNegRaw;
        int stageNegCount = 0;

        if (stageIdx == 0) {
            std::string stageNegBin;
            if (!regenerateStageNegativesFromCache(stageNegScriptPath,
                                                   stageNegCacheDir,
                                                   stageNegImagesDir,
                                                   stageNegOutDir,
                                                   stageIdx,
                                                   numNeg,
                                                   win.winW,
                                                   perStageSeed,
                                                   stageNegBin)) {
                std::cerr << "[ERR] stage negative sampling failed at stage " << stageIdx << "\n";
                return 1;
            }
            if (!loadRawSampleBin(stageNegBin, samplePixels, stageNegRaw, stageNegCount)) {
                std::cerr << "[ERR] failed to load stage negative bin: " << stageNegBin << "\n";
                return 1;
            }
            if (stageNegCount != numNeg) {
                std::cerr << "[ERR] stage0 negative count mismatch: expected " << numNeg
                          << " got " << stageNegCount << "\n";
                return 1;
            }
            std::memcpy(selectedNegRaw.data(), stageNegRaw.data(), static_cast<size_t>(numNeg) * samplePixels);
            hardSelected = numNeg;
        }

        if (stageIdx > 0 && !stages.empty()) {
            std::vector<GpuStump4> hCurStumps(selectedWeaks.size());
            for (size_t wi = 0; wi < selectedWeaks.size(); ++wi) {
                const WeakClassifier& w = selectedWeaks[wi];
                float featureBits = 0.0f;
                std::memcpy(&featureBits, &w.featureIdx, sizeof(int));
                hCurStumps[wi].st = make_float4(featureBits, w.theta, w.leftVal, w.rightVal);
            }
            CudaBuffer<GpuStump4> dCurStumps(hCurStumps.size());
            if (cudaMemcpy(dCurStumps.data(),
                           hCurStumps.data(),
                           hCurStumps.size() * sizeof(GpuStump4),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                std::cerr << "[ERR] copy mining stumps failed\n";
                return 1;
            }
            st = detectEngine.setCascadeModel(trainEngine.deviceFeaturePool(),
                                              trainEngine.featurePoolSize(),
                                              dCurStumps.data(),
                                              static_cast<int>(hCurStumps.size()),
                                              stages);
            if (st != Status::kOk) {
                std::cerr << "[ERR] set mining cascade model failed: " << statusToString(st) << "\n";
                return 1;
            }

            std::vector<std::string> minePgmPaths = collectPgmPaths(stageNegImagesDir);
            if (minePgmPaths.empty()) {
                std::cerr << "[ERR] No PGM mining images found in " << stageNegImagesDir
                          << ". Build PGM cache first via scripts/stage_negatives.py --build-pgm-cache\n";
                return 1;
            }
            std::mt19937 mineRng(static_cast<uint32_t>(perStageSeed));
            std::shuffle(minePgmPaths.begin(), minePgmPaths.end(), mineRng);
            std::vector<uint8_t> onePatch;
            const int perImageKeep = 256;
            int scannedImages = 0;

            for (const std::string& p : minePgmPaths) {
                if (hardSelected >= numNeg) break;
                GrayImage negImg;
                if (!loadPGM(p, negImg)) continue;
                if (negImg.w < win.winW || negImg.h < win.winH) continue;
                if (negImg.w > miningMaxImageW || negImg.h > miningMaxImageH) continue;
                const size_t bytes = static_cast<size_t>(negImg.w) * negImg.h;
                if (cudaMemcpy(dMiningImage.data(), negImg.data.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                    continue;
                }
                std::vector<Detection> mineDetections;
                st = detectEngine.detectMultiScale(dMiningImage.data(),
                                                   negImg.w,
                                                   negImg.h,
                                                   negImg.w,
                                                   scaleFactor,
                                                   0,
                                                   win.winW,
                                                   std::max(20000, numNeg * 2),
                                                   mineDetections,
                                                   false);
                if (st != Status::kOk) continue;
                if (!mineDetections.empty()) {
                    const int keepN = std::min(static_cast<int>(mineDetections.size()), perImageKeep);
                    std::partial_sort(mineDetections.begin(),
                                      mineDetections.begin() + keepN,
                                      mineDetections.end(),
                                      [](const Detection& a, const Detection& b) {
                                          return a.score > b.score;
                                      });
                    mineDetections.resize(static_cast<size_t>(keepN));
                }
                for (const Detection& d : mineDetections) {
                    if (hardSelected >= numNeg) break;
                    cropResizeTo24(negImg, d, onePatch);
                    std::memcpy(selectedNegRaw.data() + static_cast<size_t>(hardSelected) * samplePixels,
                                onePatch.data(),
                                samplePixels);
                    ++hardSelected;
                }
                ++scannedImages;
                if ((scannedImages % 20) == 0) {
                    std::cout << "[MINE] stage=" << stageIdx
                              << " scanned=" << scannedImages
                              << " hard=" << hardSelected << "/" << numNeg << "\n";
                }
            }

            std::cout << "[TRAIN] stage=" << stageIdx
                      << " hardNegSelected=" << hardSelected
                      << "/" << numNeg << "\n";
        }

        if (hardSelected < numNeg) {
            std::string stageNegBin;
            const int fallbackCount = std::max(numNeg, (numNeg - hardSelected) * hardNegCandidateMultiplier);
            if (!regenerateStageNegativesFromCache(stageNegScriptPath,
                                                   stageNegCacheDir,
                                                   stageNegImagesDir,
                                                   stageNegOutDir,
                                                   stageIdx,
                                                   fallbackCount,
                                                   win.winW,
                                                   perStageSeed ^ 0x9e3779b9,
                                                   stageNegBin)) {
                std::cerr << "[ERR] fallback stage negative sampling failed at stage " << stageIdx << "\n";
                return 1;
            }
            if (!loadRawSampleBin(stageNegBin, samplePixels, stageNegRaw, stageNegCount) || stageNegCount <= 0) {
                std::cerr << "[ERR] failed to load fallback stage negative bin: " << stageNegBin << "\n";
                return 1;
            }
            std::mt19937 rng(perStageSeed ^ 0x5bd1e995u);
            std::uniform_int_distribution<int> pick(0, stageNegCount - 1);
            while (hardSelected < numNeg) {
                const int srcIdx = pick(rng);
                std::memcpy(selectedNegRaw.data() + static_cast<size_t>(hardSelected) * samplePixels,
                            stageNegRaw.data() + static_cast<size_t>(srcIdx) * samplePixels,
                            samplePixels);
                ++hardSelected;
            }
        }

        std::memcpy(trainRaw.data(), selectedPosRaw.data(), selectedPosRaw.size());
        std::memcpy(trainRaw.data() + selectedPosRaw.size(), selectedNegRaw.data(), selectedNegRaw.size());
        for (int i = numPos; i < nSamples; ++i) {
            hActive[static_cast<size_t>(i)] = 1;
        }

        if (cudaMemcpy(dTrain.data(),
                       trainRaw.data(),
                       trainRaw.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(dActive.data(),
                       hActive.data(),
                       static_cast<size_t>(nSamples) * sizeof(uint8_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "[ERR] copy stage train set to device failed\n";
            return 1;
        }

        st = trainEngine.computeIntegralBatchTransposed(dTrain.data(), win.winW, win.winH, win.winW, nSamples);
        if (st != Status::kOk) {
            std::cerr << "[ERR] recompute integrals failed at stage " << stageIdx
                      << ": " << statusToString(st) << "\n";
            return 1;
        }

        std::vector<int> incomingNegIdx;
        incomingNegIdx.reserve(static_cast<size_t>(numNeg));
        for (int i = numPos; i < nSamples; ++i) {
            if (hActive[i] != 0) incomingNegIdx.push_back(i);
        }
        const int stageNegBaseCount = static_cast<int>(incomingNegIdx.size());
        if (stageNegBaseCount <= 0) {
            std::cout << "[TRAIN] No surviving negatives, stop at stage " << stageIdx << "\n";
            break;
        }

        if (!resetStageWeights(stageNegBaseCount)) {
            std::cerr << "[ERR] resetStageWeights failed\n";
            return 1;
        }
        if (!normalizeWeights()) {
            std::cerr << "[ERR] normalizeWeights failed at stage init\n";
            return 1;
        }

        if (debugEvery > 0) {
            if (cudaMemcpy(hWeightsDbg.data(),
                           dWeight.data(),
                           static_cast<size_t>(nSamples) * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::cerr << "[ERR] copy weights for debug failed at stage init\n";
                return 1;
            }
            float initPosW = 0.0f;
            float initNegW = 0.0f;
            for (int i = 0; i < numPos; ++i) initPosW += hWeightsDbg[static_cast<size_t>(i)];
            for (int i = numPos; i < nSamples; ++i) initNegW += hWeightsDbg[static_cast<size_t>(i)];
            std::cout << "[DBG] stage=" << stageIdx
                      << " initWeightSum pos=" << initPosW
                      << " neg=" << initNegW
                      << " activeNeg=" << stageNegBaseCount
                      << "\n";
        }

        if (cudaMemset(dUsedFeatureMask.data(), 0, static_cast<size_t>(numFeatures) * sizeof(uint8_t)) != cudaSuccess) {
            std::cerr << "[ERR] clear used-feature mask failed\n";
            return 1;
        }
        const int stageFirst = static_cast<int>(selectedWeaks.size());
        float stageThreshold = 0.0f;
        bool stageSatisfied = false;
        FillArrayKernel<<<grd, blk>>>(dStrong.data(), nSamples, 0.0f);
        if (cudaGetLastError() != cudaSuccess) {
            std::cerr << "[ERR] FillArrayKernel launch failed at stage init\n";
            return 1;
        }

        int stageWeakLimit = maxWeakPerStage;
        stageWeakLimit = std::min(stageWeakLimit, kPaperTotalFeatures - static_cast<int>(selectedWeaks.size()));
        if (stageWeakLimit <= 0) {
            break;
        }

        for (int weakRound = 0; weakRound < stageWeakLimit; ++weakRound) {
            float bestErr = std::numeric_limits<float>::infinity();
            float bestTheta = 0.0f;
            int bestParity = +1;
            int bestFeature = -1;

            int tileIdx = 0;
            for (int begin = 0; begin < numFeatures; begin += featureTile) {
                const int count = std::min(featureTile, numFeatures - begin);

                st = trainer.evaluateFeatureResponses(trainEngine.deviceFeaturePool(), begin, count);
                if (st != Status::kOk) {
                    std::cerr << "[ERR] evaluateFeatureResponses failed: " << statusToString(st) << "\n";
                    return 1;
                }

                st = trainer.sortSamplesPerFeature(trainer.responseBuffer(), count);
                if (st != Status::kOk) {
                    std::cerr << "[ERR] sortSamplesPerFeature failed: " << statusToString(st) << "\n";
                    return 1;
                }

                FeatureTileView tile{};
                tile.featureBegin = begin;
                tile.featureCount = count;
                tile.d_features = trainEngine.deviceFeaturePool();
                tile.d_resp = trainer.responseBuffer();
                tile.d_sortedIdx = trainer.sortedIndexBuffer();

                st = trainer.evaluateAndFindThreshold(tile);
                if (st != Status::kOk) {
                    std::cerr << "[ERR] evaluateAndFindThreshold failed: " << statusToString(st) << "\n";
                    return 1;
                }

                st = trainer.selectBestInTile(begin,
                                              count,
                                              dUsedFeatureMask.data(),
                                              dTileBestPerRound.data() + tileIdx);
                if (st != Status::kOk) {
                    std::cerr << "[ERR] selectBestInTile failed: " << statusToString(st) << "\n";
                    return 1;
                }
                ++tileIdx;
            }

            ReduceTileBestCandidatesKernel<<<1, 256>>>(dTileBestPerRound.data(), tileIdx, dRoundBest.data());
            if (cudaGetLastError() != cudaSuccess) {
                std::cerr << "[ERR] ReduceTileBestCandidatesKernel launch failed\n";
                return 1;
            }
            if (cudaMemcpy(&hTileBest,
                           dRoundBest.data(),
                           sizeof(TileBestCandidate),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::cerr << "[ERR] memcpy round-best failed\n";
                return 1;
            }
            bestErr = hTileBest.bestErr;
            bestTheta = hTileBest.bestTheta;
            bestParity = hTileBest.bestParity;
            bestFeature = hTileBest.featureIdx;

            if (!(bestFeature >= 0) || !std::isfinite(bestErr)) {
                std::cerr << "[WARN] No valid weak found at stage " << stageIdx
                          << " weakRound " << weakRound << "\n";
                break;
            }
            if (cudaMemsetAsync(dUsedFeatureMask.data() + bestFeature,
                                1,
                                sizeof(uint8_t)) != cudaSuccess) {
                std::cerr << "[ERR] mark used-feature mask failed\n";
                return 1;
            }

            const float eps = 1e-6f;
            const float errClamped = std::min(std::max(bestErr, eps), 1.0f - eps);
            const float coeffC = std::log((1.0f - errClamped) / errClamped);

            WeakClassifier weak{};
            weak.featureIdx = bestFeature;
            weak.theta = bestTheta;
            // Parity convention is unified with threshold kernel:
            // parity>0 => (resp < theta ? -1 : +1)
            // parity<0 => (resp < theta ? +1 : -1)
            weak.parity = static_cast<int8_t>(bestParity);
            weak.err = errClamped;
            weak.alpha = coeffC;
            if (weak.parity > 0) {
                weak.leftVal = -coeffC;
                weak.rightVal = +coeffC;
            } else {
                weak.leftVal = +coeffC;
                weak.rightVal = -coeffC;
            }
            selectedWeaks.push_back(weak);

            st = trainer.evaluateFeatureResponses(trainEngine.deviceFeaturePool(), bestFeature, 1);
            if (st != Status::kOk) {
                std::cerr << "[ERR] evaluateFeatureResponses(selected weak) failed\n";
                return 1;
            }

            const bool emitDebug = (debugEvery > 0) &&
                                   ((weakRound % debugEvery) == 0 || weakRound == stageWeakLimit - 1);
            if (emitDebug) {
                if (cudaMemcpy(hRespDbg.data(),
                               trainer.responseBuffer(),
                               static_cast<size_t>(nSamples) * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    std::cerr << "[ERR] copy response for parity debug failed\n";
                    return 1;
                }
                if (cudaMemcpy(hWeightsDbg.data(),
                               dWeight.data(),
                               static_cast<size_t>(nSamples) * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    std::cerr << "[ERR] copy weights for parity debug failed\n";
                    return 1;
                }
                double errP = 0.0;  // parity +1
                double errN = 0.0;  // parity -1
                for (int i = 0; i < nSamples; ++i) {
                    if (hActive[static_cast<size_t>(i)] == 0) continue;
                    const float r = hRespDbg[static_cast<size_t>(i)];
                    const int y = static_cast<int>(hLabels[static_cast<size_t>(i)]);
                    const float w = hWeightsDbg[static_cast<size_t>(i)];
                    const int predP = (r < weak.theta) ? -1 : +1;
                    const int predN = (r < weak.theta) ? +1 : -1;
                    if (predP != y) errP += static_cast<double>(w);
                    if (predN != y) errN += static_cast<double>(w);
                }
                std::cout << "[DBG] stage=" << stageIdx
                          << " weak=" << weakRound
                          << " bestParityRaw=" << bestParity
                          << " parityUsed=" << static_cast<int>(weak.parity)
                          << " errP=" << errP
                          << " errN=" << errN
                          << " theta=" << weak.theta
                          << "\n";
            }

            // Incrementally build current-stage strong score on GPU.
            AccumulateStrongScoreKernel<<<grd, blk>>>(trainer.responseBuffer(),
                                                      dStrong.data(),
                                                      nSamples,
                                                      weak.theta,
                                                      weak.leftVal,
                                                      weak.rightVal);
            if (cudaGetLastError() != cudaSuccess) {
                std::cerr << "[ERR] AccumulateStrongScoreKernel launch failed\n";
                return 1;
            }

            UpdateWeightsDiscreteKernel<<<grd, blk>>>(trainer.responseBuffer(),
                                                      dLabel.data(),
                                                      dActive.data(),
                                                      dWeight.data(),
                                                      nSamples,
                                                      weak.theta,
                                                      static_cast<int>(weak.parity),
                                                      coeffC);
            if (cudaGetLastError() != cudaSuccess) {
                std::cerr << "[ERR] UpdateWeightsDiscreteKernel launch failed\n";
                return 1;
            }
            if (!normalizeWeights()) {
                std::cerr << "[ERR] normalizeWeights failed after weak update\n";
                return 1;
            }

            if (cudaMemcpy(hStrong.data(),
                           dStrong.data(),
                           static_cast<size_t>(nSamples) * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::cerr << "[ERR] copy strong scores failed\n";
                return 1;
            }

            float hitRate = 0.0f;
            calibrateStageThreshold(hStrong, stageThreshold, hitRate);

            int negSurvive = 0;
            for (int idx : incomingNegIdx) {
                if (hStrong[static_cast<size_t>(idx)] >= stageThreshold) {
                    ++negSurvive;
                }
            }

            const float falseAlarm = static_cast<float>(negSurvive) / static_cast<float>(stageNegBaseCount);

            if (emitDebug) {
                float posMin = std::numeric_limits<float>::infinity();
                float posMax = -std::numeric_limits<float>::infinity();
                double posSum = 0.0;
                for (int i = 0; i < numPos; ++i) {
                    const float s = hStrong[static_cast<size_t>(i)];
                    posMin = std::min(posMin, s);
                    posMax = std::max(posMax, s);
                    posSum += s;
                }
                const float posMean = static_cast<float>(posSum / static_cast<double>(numPos));
                float posMedian = 0.0f;
                if (!posScores.empty()) {
                    auto midIt = posScores.begin() + (posScores.size() / 2);
                    std::nth_element(posScores.begin(), midIt, posScores.end());
                    posMedian = *midIt;
                }

                float negMin = std::numeric_limits<float>::infinity();
                float negMax = -std::numeric_limits<float>::infinity();
                double negSum = 0.0;
                for (int idx : incomingNegIdx) {
                    const float s = hStrong[static_cast<size_t>(idx)];
                    negMin = std::min(negMin, s);
                    negMax = std::max(negMax, s);
                    negSum += s;
                }
                const float negMean =
                    incomingNegIdx.empty() ? 0.0f : static_cast<float>(negSum / static_cast<double>(incomingNegIdx.size()));

                if (cudaMemcpy(hWeightsDbg.data(),
                               dWeight.data(),
                               static_cast<size_t>(nSamples) * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    std::cerr << "[ERR] copy weights for debug failed\n";
                    return 1;
                }
                float sumPosW = 0.0f;
                float sumNegW = 0.0f;
                for (int i = 0; i < numPos; ++i) sumPosW += hWeightsDbg[static_cast<size_t>(i)];
                for (int idx : incomingNegIdx) sumNegW += hWeightsDbg[static_cast<size_t>(idx)];

                std::cout << "[DBG] stage=" << stageIdx
                          << " weak=" << weakRound
                          << " score_pos[min,max,mean,p50]=[" << posMin << "," << posMax << "," << posMean << "," << posMedian << "]"
                          << " score_neg[min,max,mean]=[" << negMin << "," << negMax << "," << negMean << "]"
                          << " threshold=" << stageThreshold
                          << " weightSum[pos,neg]=[" << sumPosW << "," << sumNegW << "]"
                          << "\n";
            }

            std::cout << "[TRAIN] stage=" << stageIdx
                      << " weak=" << weakRound
                      << " feature=" << weak.featureIdx
                      << " err=" << weak.err
                      << " alpha(C)=" << weak.alpha
                      << " parity=" << static_cast<int>(weak.parity)
                      << " hit=" << hitRate
                      << " fa=" << falseAlarm
                      << " negSurvive=" << negSurvive << "/" << stageNegBaseCount
                      << "\n";

            if ((hitRate >= minHitRate && falseAlarm <= maxFalseAlarm) || negSurvive == 0) {
                stageSatisfied = true;
                break;
            }
        }

        const int stageTrees = static_cast<int>(selectedWeaks.size()) - stageFirst;
        if (stageTrees <= 0) {
            std::cerr << "[WARN] Stage " << stageIdx << " produced no weak classifiers, stop\n";
            break;
        }

        // Apply stage rejection once after stage threshold is finalized.
        if (cudaMemcpy(hStrong.data(),
                       dStrong.data(),
                       static_cast<size_t>(nSamples) * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cerr << "[ERR] copy final strong scores failed\n";
            return 1;
        }
        for (int idx : incomingNegIdx) {
            hActive[static_cast<size_t>(idx)] =
                (hStrong[static_cast<size_t>(idx)] >= stageThreshold) ? 1 : 0;
        }
        if (cudaMemcpy(dActive.data(), hActive.data(),
                       static_cast<size_t>(nSamples) * sizeof(uint8_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "[ERR] copy stage active mask failed\n";
            return 1;
        }

        int remainNeg = 0;
        for (int i = numPos; i < nSamples; ++i) {
            if (hActive[i] != 0) ++remainNeg;
        }
        const float totalFalseAlarm = static_cast<float>(remainNeg) / static_cast<float>(numNeg);

        if (!stageSatisfied && remainNeg > 0) {
            std::cerr << "[WARN] Stage " << stageIdx
                      << " did not meet target (hit>=" << minHitRate
                      << ", fa<=" << maxFalseAlarm
                      << "). Increase maxWeakPerStage or improve training data.\n";
            break;
        }

        CascadeStage cs{};
        cs.first = stageFirst;
        cs.ntrees = stageTrees;
        // Align with OpenCV load-time behavior: stage threshold uses a tiny negative epsilon.
        cs.threshold = stageThreshold - 1e-5f;
        stages.push_back(cs);

        std::cout << "[TRAIN] Stage " << stageIdx
                  << " done: ntrees=" << stageTrees
                  << " threshold=" << stageThreshold
                  << " stageOK=" << (stageSatisfied ? 1 : 0)
                  << " remainNeg=" << remainNeg << "/" << numNeg
                  << " totalFA=" << totalFalseAlarm
                  << "\n";

        if (remainNeg == 0) {
            break;
        }
    }

    cudaDeviceSynchronize();
    const auto tTrain1 = std::chrono::high_resolution_clock::now();
    const double trainMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(tTrain1 - tTrain0).count();
    std::cout << "[TRAIN] elapsed(ms): " << trainMs << "\n";

    if (selectedWeaks.empty() || stages.empty()) {
        std::cerr << "[ERR] no trained cascade stages\n";
        return 1;
    }

    std::vector<GpuStump4> hStumps(selectedWeaks.size());
    for (size_t i = 0; i < selectedWeaks.size(); ++i) {
        const WeakClassifier& w = selectedWeaks[i];
        float featureBits = 0.0f;
        std::memcpy(&featureBits, &w.featureIdx, sizeof(int));
        hStumps[i].st = make_float4(featureBits, w.theta, w.leftVal, w.rightVal);
    }

    if (!vj::ModelIO::saveCascadeModel(modelOutPath.c_str(), win, hFeatures, hStumps, stages)) {
        std::cerr << "[ERR] saveCascadeModel failed: " << modelOutPath << "\n";
        return 1;
    }
    std::cout << "[OK] Saved model to: " << modelOutPath
              << " (features=" << hFeatures.size()
              << ", stumps=" << hStumps.size()
              << ", stages=" << stages.size() << ")\n";
    return 0;
}

int runDetect(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: cuda_hello detect <in_model.bin> <image.pgm> <out.ppm>"
                  << " [scaleFactor=1.25] [minNeighbors=4] [minObjectSize=24] [maxDetections=200000]\n";
        return 1;
    }

    const std::string modelPath = argv[1];
    const std::string imagePath = argv[2];
    const std::string outPath = argv[3];

    float scaleFactor = 1.25f;
    int minNeighbors = 4;
    int minObjectSize = 24;
    int maxDetections = 200000;
    if (argc > 4) scaleFactor = std::stof(argv[4]);
    if (argc > 5) minNeighbors = std::max(1, std::stoi(argv[5]));
    if (argc > 6) minObjectSize = std::max(1, std::stoi(argv[6]));
    if (argc > 7) maxDetections = std::max(100, std::stoi(argv[7]));

    WindowSpec win{};
    std::vector<HaarFeature> hFeatures;
    std::vector<GpuStump4> hStumps;
    std::vector<CascadeStage> hStages;
    if (!vj::ModelIO::loadCascadeModel(modelPath.c_str(), win, hFeatures, hStumps, hStages)) {
        std::cerr << "[ERR] loadCascadeModel failed: " << modelPath << "\n";
        return 1;
    }

    GrayImage img;
    if (!loadPGM(imagePath, img)) {
        return 1;
    }
    const int maxImageW = std::max(win.winW, img.w);
    const int maxImageH = std::max(win.winH, img.h);

    FaceVisionEngine detectEngine(win, 1, maxImageW, maxImageH, false, maxDetections);
    Status st = detectEngine.uploadFeaturePool(hFeatures);
    if (st != Status::kOk) {
        std::cerr << "[ERR] uploadFeaturePool failed: " << statusToString(st) << "\n";
        return 1;
    }

    CudaBuffer<GpuStump4> dStumps(hStumps.size());
    CudaBuffer<uint8_t> dImage(static_cast<size_t>(img.w) * img.h);
    if (cudaMemcpy(dStumps.data(),
                   hStumps.data(),
                   hStumps.size() * sizeof(GpuStump4),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dImage.data(),
                   img.data.data(),
                   static_cast<size_t>(img.w) * img.h,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "[ERR] cudaMemcpy to device failed\n";
        return 1;
    }

    st = detectEngine.setCascadeModel(detectEngine.deviceFeaturePool(),
                                      detectEngine.featurePoolSize(),
                                      dStumps.data(),
                                      static_cast<int>(hStumps.size()),
                                      hStages);
    if (st != Status::kOk) {
        std::cerr << "[ERR] setCascadeModel failed: " << statusToString(st) << "\n";
        return 1;
    }

    std::vector<Detection> detections;
    const auto tDetect0 = std::chrono::high_resolution_clock::now();
    st = detectEngine.detectMultiScale(dImage.data(),
                                       img.w,
                                       img.h,
                                       img.w,
                                       scaleFactor,
                                       minNeighbors,
                                       minObjectSize,
                                       maxDetections,
                                       detections,
                                       true);
    if (st != Status::kOk) {
        std::cerr << "[ERR] detectMultiScale failed: " << statusToString(st) << "\n";
        return 1;
    }
    const auto tDetect1 = std::chrono::high_resolution_clock::now();
    const double detectMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(tDetect1 - tDetect0).count();
    std::cout << "[DETECT] detections=" << detections.size() << " elapsed(ms): " << detectMs << "\n";

    std::vector<uint8_t> rgb(static_cast<size_t>(img.w) * img.h * 3);
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            const uint8_t g = img.data[static_cast<size_t>(y) * img.w + x];
            const size_t idx = (static_cast<size_t>(y) * img.w + x) * 3;
            rgb[idx + 0] = g;
            rgb[idx + 1] = g;
            rgb[idx + 2] = g;
        }
    }
    for (const Detection& d : detections) {
        drawRect(rgb, img.w, img.h, d);
    }
    if (!savePPM(outPath, img.w, img.h, rgb)) {
        std::cerr << "[ERR] savePPM failed\n";
        return 1;
    }
    std::cout << "[OK] Saved result to: " << outPath << "\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " train ... <out_model.bin>\n";
        std::cerr << "   or: " << argv[0] << " detect <in_model.bin> <image.pgm> <out.ppm>\n";
        return 1;
    }
    const std::string mode = argv[1];
    if (mode == "train") {
        return runTrain(argc - 1, argv + 1);
    }
    if (mode == "detect") {
        return runDetect(argc - 1, argv + 1);
    }
    std::cerr << "[ERR] Unknown mode: " << mode << "\n";
    std::cerr << "Usage: " << argv[0] << " train ... <out_model.bin>\n";
    std::cerr << "   or: " << argv[0] << " detect <in_model.bin> <image.pgm> <out.ppm>\n";
    return 1;
}
