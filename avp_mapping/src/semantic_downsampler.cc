#include "avp_mapping/semantic_downsampler.h"

#include <cmath>

namespace avp_mapping {

namespace {
SemanticDownsampler::Config g_config;
SemanticDownsampler g_downsampler(g_config);
}  // namespace

SemanticDownsampler::SemanticDownsampler(const Config &cfg)
    : voxel_size_m_(cfg.voxel_size) {}

Eigen::Vector3i SemanticDownsampler::Quantize(const Eigen::Vector3i &grid_pt) const {
  if (voxel_size_m_ <= kPixelScale || voxel_size_m_ <= 0.0) {
    return grid_pt;
  }

  Eigen::Vector3d world = grid_pt.cast<double>() * kPixelScale;
  Eigen::Vector3d quantized_world =
      (world.array() / voxel_size_m_).round() * voxel_size_m_;
  Eigen::Vector3i quantized;
  quantized.x() = static_cast<int>(std::llround(quantized_world.x() * kPixelScaleInv));
  quantized.y() = static_cast<int>(std::llround(quantized_world.y() * kPixelScaleInv));
  quantized.z() = static_cast<int>(std::llround(quantized_world.z() * kPixelScaleInv));
  return quantized;
}

SemanticDownsampler &GetSemanticDownsampler() { return g_downsampler; }

void SetSemanticDownsamplerConfig(const SemanticDownsampler::Config &config) {
  g_config = config;
  g_downsampler = SemanticDownsampler(config);
}

}  // namespace avp_mapping
