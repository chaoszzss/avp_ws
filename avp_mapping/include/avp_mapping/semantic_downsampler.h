#pragma once

#include <Eigen/Core>
#include <unordered_set>

#include "avp_mapping/slot.h"

namespace avp_mapping {

class SemanticDownsampler {
 public:
  struct Config {
    double voxel_size{0.1};  // meters, should be >= kPixelScale
  };

  explicit SemanticDownsampler(const Config &cfg);

  Eigen::Vector3i Quantize(const Eigen::Vector3i &grid_pt) const;

 private:
  double voxel_size_m_{0.05};
};

SemanticDownsampler &GetSemanticDownsampler();
void SetSemanticDownsamplerConfig(const SemanticDownsampler::Config &config);

}  // namespace avp_mapping
