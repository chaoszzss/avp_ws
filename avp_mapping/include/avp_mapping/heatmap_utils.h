#pragma once

#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "avp_mapping/map.h"

namespace avp_mapping {
namespace heatmap {

struct Extent2D {
  double min_x{0.0};
  double max_x{0.0};
  double min_y{0.0};
  double max_y{0.0};

  bool isValid() const { return max_x > min_x && max_y > min_y; }
  double width() const { return max_x - min_x; }
  double height() const { return max_y - min_y; }
  void expand(double padding_m) {
    min_x -= padding_m;
    max_x += padding_m;
    min_y -= padding_m;
    max_y += padding_m;
  }
};

struct SemanticHeatmapResult {
  cv::Mat counts;        // CV_32F grid of unique label counts
  cv::Mat visualization; // CV_8UC3 heatmap image
  Extent2D extent;
  double resolution{0.5};
};

struct GeometricHeatmapResult {
  cv::Mat density;       // CV_32F grid of point counts
  cv::Mat visualization; // CV_8UC3 heatmap image
  Extent2D extent;
  double resolution{0.5};
};

Extent2D ComputeSemanticExtent(const Map &map, double padding_m = 1.0);
Extent2D ComputeGeometricExtent(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
                                double padding_m = 1.0);
Extent2D CombineExtents(const Extent2D &a, const Extent2D &b);

SemanticHeatmapResult RenderSemanticCoverageHeatmap(
    const Map &map, const Extent2D &extent, double resolution_m,
    int colormap = cv::COLORMAP_TURBO);

GeometricHeatmapResult RenderGeometricDensityHeatmap(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud, const Extent2D &extent,
    double resolution_m, bool log_scale = true,
    int colormap = cv::COLORMAP_VIRIDIS);

} // namespace heatmap
} // namespace avp_mapping
