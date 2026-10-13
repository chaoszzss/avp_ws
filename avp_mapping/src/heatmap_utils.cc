#include "avp_mapping/heatmap_utils.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace avp_mapping {
namespace heatmap {
namespace {

template <typename Fn>
Extent2D ComputeExtentFromIterator(Fn &&visit, double padding_m) {
  Extent2D extent;
  bool has_value = false;
  visit([&](double x, double y) {
    if (!std::isfinite(x) || !std::isfinite(y)) {
      return;
    }
    if (!has_value) {
      extent.min_x = extent.max_x = x;
      extent.min_y = extent.max_y = y;
      has_value = true;
    } else {
      extent.min_x = std::min(extent.min_x, x);
      extent.max_x = std::max(extent.max_x, x);
      extent.min_y = std::min(extent.min_y, y);
      extent.max_y = std::max(extent.max_y, y);
    }
  });

  if (!has_value) {
    return extent;
  }

  extent.expand(padding_m);
  return extent;
}

cv::Mat ApplyColorMapWithZeroMask(const cv::Mat &input_norm, const cv::Mat &zero_mask,
                                  int colormap) {
  cv::Mat colored;
  cv::applyColorMap(input_norm, colored, colormap);
  if (!zero_mask.empty()) {
    colored.setTo(cv::Scalar::all(0), zero_mask);
  }
  return colored;
}

cv::Mat NormalizeForHeatmap(const cv::Mat &values, bool log_scale) {
  cv::Mat processed;
  if (log_scale) {
    cv::Mat temp;
    values.convertTo(temp, CV_32F);
    cv::Mat offset = temp + 1.0f;
    cv::log(offset, processed);
  } else {
    values.convertTo(processed, CV_32F);
  }

  double min_v = 0.0;
  double max_v = 0.0;
  cv::minMaxLoc(processed, &min_v, &max_v);
  if (max_v <= 0.0) {
    max_v = 1.0;
  }
  cv::Mat normalized;
  processed.convertTo(normalized, CV_8U, 255.0 / max_v, 0.0);
  return normalized;
}

cv::Mat UpsampleNearest(const cv::Mat &image, double factor) {
  if (factor <= 1.0) {
    return image.clone();
  }
  cv::Mat enlarged;
  cv::resize(image, enlarged, cv::Size(), factor, factor, cv::INTER_NEAREST);
  return enlarged;
}

}  // namespace

Extent2D ComputeSemanticExtent(const Map &map, double padding_m) {
  auto extent = ComputeExtentFromIterator(
      [&](const auto &emit) {
        const auto bbox = map.getBoundingBox();
        if (!bbox.isEmpty()) {
          emit(bbox.min().x(), bbox.min().y());
          emit(bbox.max().x(), bbox.max().y());
        }

        const auto slots = map.getAllSlots();
        for (const auto &slot : slots) {
          for (const auto &corner : slot.corners_) {
            emit(corner.x(), corner.y());
          }
        }

        for (int label = 0; label < SemanticLabel::kTotalLabelNum; ++label) {
          const auto &elements = map.getSemanticElement(
              static_cast<SemanticLabel>(label));
          for (const auto &pt : elements) {
            emit(pt.x() * kPixelScale, pt.y() * kPixelScale);
          }
        }
      },
      padding_m);
  return extent;
}

Extent2D ComputeGeometricExtent(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
                                double padding_m) {
  if (!cloud || cloud->empty()) {
    return {};
  }

  auto extent = ComputeExtentFromIterator(
      [&](const auto &emit) {
        for (const auto &pt : cloud->points) {
          emit(static_cast<double>(pt.x), static_cast<double>(pt.y));
        }
      },
      padding_m);
  return extent;
}

Extent2D CombineExtents(const Extent2D &a, const Extent2D &b) {
  if (!a.isValid()) {
    return b;
  }
  if (!b.isValid()) {
    return a;
  }
  Extent2D combined;
  combined.min_x = std::min(a.min_x, b.min_x);
  combined.max_x = std::max(a.max_x, b.max_x);
  combined.min_y = std::min(a.min_y, b.min_y);
  combined.max_y = std::max(a.max_y, b.max_y);
  return combined;
}

SemanticHeatmapResult RenderSemanticCoverageHeatmap(
    const Map &map, const Extent2D &extent, double resolution_m, int colormap) {
  SemanticHeatmapResult result;
  result.extent = extent;
  result.resolution = resolution_m;
  if (!extent.isValid() || resolution_m <= 0.0) {
    return result;
  }

  const int cols = static_cast<int>(std::ceil(extent.width() / resolution_m));
  const int rows = static_cast<int>(std::ceil(extent.height() / resolution_m));
  if (cols <= 0 || rows <= 0) {
    return result;
  }

  result.counts = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
  std::vector<std::bitset<SemanticLabel::kTotalLabelNum>> bitsets(rows * cols);

  auto accumulate = [&](double x, double y, SemanticLabel label) {
    const int col =
        static_cast<int>(std::floor((x - extent.min_x) / resolution_m));
    const int row =
        static_cast<int>(std::floor((extent.max_y - y) / resolution_m));
    if (col < 0 || col >= cols || row < 0 || row >= rows) {
      return;
    }
    bitsets[row * cols + col].set(static_cast<size_t>(label));
  };

  const auto slots = map.getAllSlots();
  for (const auto &slot : slots) {
    for (const auto &corner : slot.corners_) {
      accumulate(corner.x(), corner.y(), SemanticLabel::kSlot);
    }
  }

  for (int label = 0; label < SemanticLabel::kTotalLabelNum; ++label) {
    const auto &elements =
        map.getSemanticElement(static_cast<SemanticLabel>(label));
    for (const auto &pt : elements) {
      accumulate(pt.x() * kPixelScale, pt.y() * kPixelScale,
                 static_cast<SemanticLabel>(label));
    }
  }

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const auto &bits = bitsets[r * cols + c];
      result.counts.at<float>(r, c) = static_cast<float>(bits.count());
    }
  }

  cv::Mat normalized = NormalizeForHeatmap(result.counts, false);
  cv::Mat zero_mask = result.counts <= 0.0f;
  const double upscale = std::max(1.0, std::min(4.0, 1.0 / resolution_m));
  result.visualization =
      UpsampleNearest(ApplyColorMapWithZeroMask(normalized, zero_mask, colormap),
                      upscale);
  return result;
}

GeometricHeatmapResult RenderGeometricDensityHeatmap(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud, const Extent2D &extent,
    double resolution_m, bool log_scale, int colormap) {
  GeometricHeatmapResult result;
  result.extent = extent;
  result.resolution = resolution_m;
  if (!extent.isValid() || !cloud || cloud->empty() || resolution_m <= 0.0) {
    return result;
  }

  const int cols = static_cast<int>(std::ceil(extent.width() / resolution_m));
  const int rows = static_cast<int>(std::ceil(extent.height() / resolution_m));
  if (cols <= 0 || rows <= 0) {
    return result;
  }

  result.density = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
  for (const auto &pt : cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) {
      continue;
    }
    const int col =
        static_cast<int>(std::floor((pt.x - extent.min_x) / resolution_m));
    const int row =
        static_cast<int>(std::floor((extent.max_y - pt.y) / resolution_m));
    if (col < 0 || col >= cols || row < 0 || row >= rows) {
      continue;
    }
    result.density.at<float>(row, col) += 1.0f;
  }

  cv::Mat normalized = NormalizeForHeatmap(result.density, log_scale);
  cv::Mat zero_mask = result.density <= 0.0f;
  const double upscale = std::max(1.0, std::min(4.0, 1.0 / resolution_m));
  result.visualization =
      UpsampleNearest(ApplyColorMapWithZeroMask(normalized, zero_mask, colormap),
                      upscale);
  return result;
}

}  // namespace heatmap
}  // namespace avp_mapping
