#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

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
      "Quantitative_indicators/save_num"};
  double padding{1.0};
};

bool ParseArguments(int argc, char **argv, Options &opts) {
  std::unordered_map<std::string, std::string *> flags{
      {"--semantic_map", &opts.semantic_map_path},
      {"--geometric_map", &opts.geometric_map_path},
      {"--output_dir", &opts.output_dir},
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto it = flags.find(arg);
    if (it != flags.end()) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << arg << "\n";
        return false;
      }
      *it->second = argv[++i];
      continue;
    }
    if (arg == "--padding") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --padding\n";
        return false;
      }
      opts.padding = std::stod(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: storage_metrics_tool --semantic_map path "
                   "--geometric_map path [--output_dir dir] "
                   "[--padding meters]\n";
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
  return true;
}

void EnsureDir(const std::string &dir) {
  if (dir.empty()) {
    return;
  }
  const std::string command = "mkdir -p \"" + dir + "\"";
  (void)std::system(command.c_str());
}

size_t GetFileSize(const std::string &path) {
  struct stat sb;
  if (stat(path.c_str(), &sb) != 0) {
    return 0u;
  }
  return static_cast<size_t>(sb.st_size);
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
    std::cerr << "Unable to compute spatial extent.\n";
    return 1;
  }
  const double area = extent.width() * extent.height();

  // Semantic counts
  const size_t dash_count =
      semantic_map.getSemanticElement(SemanticLabel::kDashLine).size();
  const size_t arrow_count =
      semantic_map.getSemanticElement(SemanticLabel::kArrowLine).size();
  const size_t slot_corner_count =
      semantic_map.getAllSlots().size() * 4;
  const size_t semantic_total = dash_count + arrow_count + slot_corner_count;

  const size_t point_count = cloud->size();

  const size_t geom_bytes =
      GetFileSize(options.geometric_map_path);
  const size_t semantic_bytes =
      GetFileSize(options.semantic_map_path);

  const double geom_density = (area > 0.0) ? point_count / area : 0.0;
  const double semantic_density = (area > 0.0) ? semantic_total / area : 0.0;
  const double compression_ratio =
      (geom_bytes > 0) ? static_cast<double>(semantic_bytes) /
                             static_cast<double>(geom_bytes)
                       : 0.0;

  EnsureDir(options.output_dir);

  const std::string csv_path = options.output_dir + "/storage_metrics.csv";
  std::ofstream ofs(csv_path);
  if (!ofs.is_open()) {
    std::cerr << "Unable to open " << csv_path << " for writing.\n";
    return 1;
  }

  ofs << "metric,geometric,semantic\n";
  ofs << "element_count," << point_count << ',' << semantic_total << "\n";
  ofs << "dash_count,," << dash_count << "\n";
  ofs << "arrow_count,," << arrow_count << "\n";
  ofs << "slot_corner_count,," << slot_corner_count << "\n";
  ofs << "file_size_bytes," << geom_bytes << ',' << semantic_bytes << "\n";
  ofs << "scene_area_m2," << area << ',' << area << "\n";
  ofs << "density_per_m2," << geom_density << ',' << semantic_density << "\n";
  ofs << "semantic_to_geometric_size_ratio,," << compression_ratio << "\n";

  ofs.close();

  std::cout << "Storage metrics written to " << csv_path << "\n";
  return 0;
}

