#pragma once

#include <vector>

#include "common_types.cuh"

namespace vj {
namespace ModelIO {

bool saveCascadeModel(const char* filepath,
                      const WindowSpec& win,
                      const std::vector<HaarFeature>& features,
                      const std::vector<GpuStump4>& stumps,
                      const std::vector<CascadeStage>& stages);

bool loadCascadeModel(const char* filepath,
                      WindowSpec& win,
                      std::vector<HaarFeature>& features,
                      std::vector<GpuStump4>& stumps,
                      std::vector<CascadeStage>& stages);

} // namespace ModelIO
} // namespace vj

