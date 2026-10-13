#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

#include <opencv2/imgcodecs.hpp>
#include <pcl/io/pcd_io.h>

#include "avp_mapping/heatmap_utils.h"

namespace {

struct Options {
  std::string semantic_map_path;
  std::string geometric_map_path;
  std::string output_dir{"output"};
  double resolution{0.5};
  double padding{1.0};
  bool log_density{true};
};

void PrintUsage() {
  std::cout << "Usage: map_heatmap_tool "
               "[--semantic_map path] [--geometric_map path] "
               "[--output_dir dir] [--resolution meters] [--padding meters] "
               "[--no-log-density]\n";
}

bool ParseArguments(int argc, char **argv, Options &options) {
  std::unordered_map<std::string, std::string *> string_flags{
      {"--semantic_map", &options.semantic_map_path},
      {"--geometric_map", &options.geometric_map_path},
      {"--output_dir", &options.output_dir}};

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto it_str = string_flags.find(arg);
    if (it_str != string_flags.end()) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for flag: " << arg << "\n";
        return false;
      }
      *(it_str->second) = argv[++i];
      continue;
    }

    if (arg == "--resolution") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --resolution\n";
        return false;
      }
      options.resolution = std::stod(argv[++i]);
      continue;
    }

    if (arg == "--padding") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --padding\n";
        return false;
      }
      options.padding = std::stod(argv[++i]);
      continue;
    }

    if (arg == "--no-log-density") {
      options.log_density = false;
      continue;
    }

    if (arg == "--help") {
      PrintUsage();
      std::exit(0);
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    PrintUsage();
    return false;
  }
  return true;
}

std::string JoinPath(const std::string &dir, const std::string &filename) {
  if (dir.empty()) {
    return filename;
  }
  if (dir.back() == '/') {
    return dir + filename;
  }
  return dir + "/" + filename;
}

void EnsureOutputDir(const std::string &dir) {
  if (dir.empty()) {
    return;
  }
  const std::string command = "mkdir -p \"" + dir + "\"";
  (void)std::system(command.c_str());
}

}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!ParseArguments(argc, argv, options)) {
    return 1;
  }

  if (options.semantic_map_path.empty() &&
      options.geometric_map_path.empty()) {
    std::cerr << "At least one of --semantic_map or --geometric_map must be "
                 "provided.\n";
    PrintUsage();
    return 1;
  }

  EnsureOutputDir(options.output_dir);

  avp_mapping::heatmap::Extent2D semantic_extent;
  Map semantic_map;
  bool semantic_loaded = false;

  if (!options.semantic_map_path.empty()) {
    semantic_map.load(options.semantic_map_path);
    semantic_extent =
        avp_mapping::heatmap::ComputeSemanticExtent(semantic_map, options.padding);
    semantic_loaded = semantic_extent.isValid();
    if (!semantic_loaded) {
      std::cerr << "Semantic map loaded but contains no elements.\n";
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  avp_mapping::heatmap::Extent2D geometric_extent;
  bool geometric_loaded = false;
  if (!options.geometric_map_path.empty()) {
    if (pcl::io::loadPCDFile(options.geometric_map_path, *cloud) == 0) {
      geometric_extent = avp_mapping::heatmap::ComputeGeometricExtent(
          cloud, options.padding);
      geometric_loaded = geometric_extent.isValid();
      if (!geometric_loaded) {
        std::cerr << "Geometric map loaded but contains no points.\n";
      }
    } else {
      std::cerr << "Failed to read geometric map: "
                << options.geometric_map_path << "\n";
    }
  }

  avp_mapping::heatmap::Extent2D unified_extent =
      avp_mapping::heatmap::CombineExtents(semantic_extent, geometric_extent);
  if (!unified_extent.isValid()) {
    std::cerr << "Unable to determine extent from provided data.\n";
    return 1;
  }

  if (semantic_loaded) {
    auto semantic_heatmap =
        avp_mapping::heatmap::RenderSemanticCoverageHeatmap(semantic_map,
                                                            unified_extent,
                                                            options.resolution);
    const std::string img_path =
        JoinPath(options.output_dir, "semantic_coverage_heatmap.png");
    cv::imwrite(img_path, semantic_heatmap.visualization);

    const std::string raw_path =
        JoinPath(options.output_dir, "semantic_coverage_counts.yml");
    cv::FileStorage fs(raw_path, cv::FileStorage::WRITE);
    fs << "resolution_m" << semantic_heatmap.resolution;
    fs << "extent_min_x" << semantic_heatmap.extent.min_x;
    fs << "extent_max_x" << semantic_heatmap.extent.max_x;
    fs << "extent_min_y" << semantic_heatmap.extent.min_y;
    fs << "extent_max_y" << semantic_heatmap.extent.max_y;
    fs << "counts" << semantic_heatmap.counts;
    fs.release();

    std::cout << "Semantic heatmap saved to " << img_path << "\n";
  }

  if (geometric_loaded) {
    auto geometric_heatmap = avp_mapping::heatmap::RenderGeometricDensityHeatmap(
        cloud, unified_extent, options.resolution, options.log_density);
    const std::string img_path =
        JoinPath(options.output_dir, "geometric_density_heatmap.png");
    cv::imwrite(img_path, geometric_heatmap.visualization);

    const std::string raw_path =
        JoinPath(options.output_dir, "geometric_density_counts.yml");
    cv::FileStorage fs(raw_path, cv::FileStorage::WRITE);
    fs << "resolution_m" << geometric_heatmap.resolution;
    fs << "extent_min_x" << geometric_heatmap.extent.min_x;
    fs << "extent_max_x" << geometric_heatmap.extent.max_x;
    fs << "extent_min_y" << geometric_heatmap.extent.min_y;
    fs << "extent_max_y" << geometric_heatmap.extent.max_y;
    fs << "density" << geometric_heatmap.density;
    fs.release();

    std::cout << "Geometric heatmap saved to " << img_path << "\n";
  }

  return 0;
}
