#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/io/pcd_io.h>

#include "avp_mapping/heatmap_utils.h"
#include "avp_mapping/map.h"
#include "avp_mapping/slot.h"

namespace {

struct Options {
  std::string semantic_map_path;
  std::string geometric_map_path;
  std::string output_dir{
      "/home/chaoszzss/avp_ws/src/avp_mapping/src/AVPMapping/output/"
      "Quantitative_indicators"};
  double resolution{0.2};
  double padding{1.0};
  float geometry_threshold{0.0f};
};

bool ParseArguments(int argc, char **argv, Options &opts) {
  std::unordered_map<std::string, std::string *> string_flags{
      {"--semantic_map", &opts.semantic_map_path},
      {"--geometric_map", &opts.geometric_map_path},
      {"--output_dir", &opts.output_dir},
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto it = string_flags.find(arg);
    if (it != string_flags.end()) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for flag " << arg << "\n";
        return false;
      }
      *it->second = argv[++i];
      continue;
    }
    if (arg == "--resolution") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --resolution\n";
        return false;
      }
      opts.resolution = std::stod(argv[++i]);
    } else if (arg == "--padding") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --padding\n";
        return false;
      }
      opts.padding = std::stod(argv[++i]);
    } else if (arg == "--geometry_threshold") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --geometry_threshold\n";
        return false;
      }
      opts.geometry_threshold = static_cast<float>(std::stod(argv[++i]));
    } else if (arg == "--help") {
      std::cout << "Usage: coverage_metrics_tool --semantic_map path "
                   "--geometric_map path [--output_dir dir] "
                   "[--resolution meters] [--padding meters] "
                   "[--geometry_threshold t]\n";
      return false;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  if (opts.semantic_map_path.empty() || opts.geometric_map_path.empty()) {
    std::cerr << "Both --semantic_map and --geometric_map are required.\n";
    return false;
  }
  if (opts.resolution <= 0.0) {
    std::cerr << "Resolution must be positive.\n";
    return false;
  }
  return true;
}

void EnsureDir(const std::string &dir) {
  if (dir.empty()) {
    return;
  }
  const std::string command = "mkdir -p \"" + dir + "\"";
  (void)std::system(command.c_str());
}

inline cv::Point WorldToPixel(const Eigen::Vector3d &pt,
                              const avp_mapping::heatmap::Extent2D &extent,
                              double resolution, int cols, int rows) {
  const double u = (pt.x() - extent.min_x) / resolution;
  const double v = (extent.max_y - pt.y()) / resolution;
  const int x = static_cast<int>(std::round(u));
  const int y = static_cast<int>(std::round(v));
  return {std::min(std::max(0, x), cols - 1),
          std::min(std::max(0, y), rows - 1)};
}

inline cv::Point WorldToPixel(const Eigen::Vector3i &grid,
                              const avp_mapping::heatmap::Extent2D &extent,
                              double resolution, int cols, int rows) {
  Eigen::Vector3d world(
      grid.x() * kPixelScale,
      grid.y() * kPixelScale,
      grid.z() * kPixelScale);
  return WorldToPixel(world, extent, resolution, cols, rows);
}

double MaskArea(const cv::Mat &mask, double resolution) {
  const double cell_area = resolution * resolution;
  const int cells = cv::countNonZero(mask);
  return static_cast<double>(cells) * cell_area;
}

struct CoverageResult {
  std::string label;
  double semantic_area{0.0};
  double reference_area{0.0};
  double ratio{0.0};
};

void WriteCoverageCsv(const std::string &path,
                      const std::vector<CoverageResult> &results) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    std::cerr << "Unable to open " << path << " for writing.\n";
    return;
  }
  ofs << "label,semantic_area_m2,reference_area_m2,coverage_ratio\n";
  for (const auto &r : results) {
    ofs << r.label << ","
        << r.semantic_area << ","
        << r.reference_area << ","
        << r.ratio << "\n";
  }
  ofs.close();
}

}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!ParseArguments(argc, argv, options)) {
    return 1;
  }

  Map semantic_map;
  semantic_map.load(options.semantic_map_path);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile(options.geometric_map_path, *cloud) != 0) {
    std::cerr << "Failed to load geometric map from "
              << options.geometric_map_path << "\n";
    return 1;
  }

  auto semantic_extent =
      avp_mapping::heatmap::ComputeSemanticExtent(semantic_map,
                                                  options.padding);
  auto geometric_extent =
      avp_mapping::heatmap::ComputeGeometricExtent(cloud, options.padding);
  auto extent =
      avp_mapping::heatmap::CombineExtents(semantic_extent, geometric_extent);
  if (!extent.isValid()) {
    std::cerr << "Unable to compute extent from inputs.\n";
    return 1;
  }

  const int cols =
      static_cast<int>(std::ceil(extent.width() / options.resolution));
  const int rows =
      static_cast<int>(std::ceil(extent.height() / options.resolution));
  if (cols <= 0 || rows <= 0) {
    std::cerr << "Computed grid size invalid.\n";
    return 1;
  }

  EnsureDir(options.output_dir);

  // Geometry reference mask
  auto geom_heatmap = avp_mapping::heatmap::RenderGeometricDensityHeatmap(
      cloud, extent, options.resolution, false);
  cv::Mat geom_mask = cv::Mat::zeros(rows, cols, CV_8UC1);
  if (!geom_heatmap.density.empty()) {
    cv::Mat density = geom_heatmap.density;
    cv::Mat mask = density > options.geometry_threshold;
    mask.convertTo(geom_mask, CV_8UC1, 255);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::morphologyEx(geom_mask, geom_mask, cv::MORPH_CLOSE, kernel);
  }

  const double reference_area = MaskArea(geom_mask, options.resolution);

  std::vector<CoverageResult> results;
  cv::Mat semantic_union(rows, cols, CV_8UC1, cv::Scalar(0));

  // Slots
  {
    cv::Mat slot_mask(rows, cols, CV_8UC1, cv::Scalar(0));
    const auto slots = semantic_map.getAllSlots();
    for (const auto &slot : slots) {
      std::vector<cv::Point> polygon(4);
      for (int i = 0; i < 4; ++i) {
        polygon[i] = WorldToPixel(slot.corners_[i], extent,
                                  options.resolution, cols, rows);
      }
      cv::fillConvexPoly(slot_mask, polygon, cv::Scalar(255));
    }
    semantic_union |= slot_mask;
    const double area = MaskArea(slot_mask, options.resolution);
    double ratio = (reference_area > 0.0) ? area / reference_area : 0.0;
    results.push_back({"slot_area", area, reference_area, ratio});
    const std::string img_path =
        options.output_dir + "/slot_coverage_mask.png";
    cv::imwrite(img_path, slot_mask);
  }

  // Dash lines
  {
    cv::Mat dash_mask(rows, cols, CV_8UC1, cv::Scalar(0));
    const auto &dash_points =
        semantic_map.getSemanticElement(SemanticLabel::kDashLine);
    for (const auto &pt : dash_points) {
      const cv::Point pixel = WorldToPixel(pt, extent, options.resolution,
                                           cols, rows);
      dash_mask.at<uint8_t>(pixel.y, pixel.x) = 255;
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::dilate(dash_mask, dash_mask, kernel);
    semantic_union |= dash_mask;
    const double area = MaskArea(dash_mask, options.resolution);
    double ratio = (reference_area > 0.0) ? area / reference_area : 0.0;
    results.push_back({"dashline_area", area, reference_area, ratio});
    const std::string img_path =
        options.output_dir + "/dashline_coverage_mask.png";
    cv::imwrite(img_path, dash_mask);
  }

  // Arrows
  {
    cv::Mat arrow_mask(rows, cols, CV_8UC1, cv::Scalar(0));
    const auto &arrow_points =
        semantic_map.getSemanticElement(SemanticLabel::kArrowLine);
    for (const auto &pt : arrow_points) {
      const cv::Point pixel = WorldToPixel(pt, extent, options.resolution,
                                           cols, rows);
      arrow_mask.at<uint8_t>(pixel.y, pixel.x) = 255;
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {5, 5});
    cv::dilate(arrow_mask, arrow_mask, kernel);
    semantic_union |= arrow_mask;
    const double area = MaskArea(arrow_mask, options.resolution);
    double ratio = (reference_area > 0.0) ? area / reference_area : 0.0;
    results.push_back({"arrow_area", area, reference_area, ratio});
    const std::string img_path =
        options.output_dir + "/arrow_coverage_mask.png";
    cv::imwrite(img_path, arrow_mask);
  }

  {
    const double union_area = MaskArea(semantic_union, options.resolution);
    double ratio = (reference_area > 0.0) ? union_area / reference_area : 0.0;
    results.push_back({"semantic_union", union_area, reference_area, ratio});
    const std::string img_path =
        options.output_dir + "/semantic_union_mask.png";
    cv::imwrite(img_path, semantic_union);
  }

  {
    const std::string geom_img =
        options.output_dir + "/geometry_reference_mask.png";
    cv::imwrite(geom_img, geom_mask);
  }

  const std::string csv_path =
      options.output_dir + "/coverage_metrics.csv";
  WriteCoverageCsv(csv_path, results);

  std::cout << "Coverage metrics exported to " << options.output_dir << "\n";
  return 0;
}
