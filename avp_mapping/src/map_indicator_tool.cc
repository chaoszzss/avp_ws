#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "avp_mapping/heatmap_utils.h"
#include "avp_mapping/map.h"

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string point_cloud_path;
  std::string semantic_map_path;
  std::string output_dir{"output/new_indicators"};
  double resolution{0.2};
  double padding{1.0};
  double overlap_split_ratio{0.5};
  double max_xy_range{500.0};  // filter out extreme outliers; <=0 disables
};

void PrintUsage() {
  std::cout << "Usage: map_indicator_tool --point_cloud path "
               "[--semantic_map path] [--output_dir dir] "
               "[--resolution meters] [--padding meters] "
               "[--overlap_split_ratio 0-1] [--max_xy_range meters]\n";
}

bool ParseArguments(int argc, char **argv, Options &opts) {
  std::unordered_map<std::string, std::string *> string_flags{
      {"--point_cloud", &opts.point_cloud_path},
      {"--semantic_map", &opts.semantic_map_path},
      {"--output_dir", &opts.output_dir},
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto it = string_flags.find(arg);
    if (it != string_flags.end()) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << arg << "\n";
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
    } else if (arg == "--overlap_split_ratio") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --overlap_split_ratio\n";
        return false;
      }
      opts.overlap_split_ratio = std::stod(argv[++i]);
    } else if (arg == "--max_xy_range") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --max_xy_range\n";
        return false;
      }
      opts.max_xy_range = std::stod(argv[++i]);
    } else if (arg == "--help") {
      PrintUsage();
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (opts.point_cloud_path.empty()) {
    std::cerr << "Point cloud path is required (--point_cloud).\n";
    return false;
  }
  if (opts.resolution <= 0.0) {
    std::cerr << "Resolution must be positive.\n";
    return false;
  }
  if (opts.overlap_split_ratio <= 0.0 || opts.overlap_split_ratio >= 1.0) {
    std::cerr << "overlap_split_ratio must be in (0,1).\n";
    return false;
  }
  return true;
}

std::string ResolvePointCloudPath(const std::string &input_path) {
  fs::path p(input_path);
  if (!fs::exists(p)) {
    return {};
  }

  if (fs::is_regular_file(p)) {
    return p.string();
  }

  if (!fs::is_directory(p)) {
    return {};
  }

  // Prefer known filenames if a directory is provided.
  const std::vector<std::string> preferred{
      "cloudGlobal.pcd",
      "cloudSurf.pcd",
      "cloudCorner.pcd",
      "custom_map_scaled_shifted.pcd",
      "custom_map_scaled.pcd",
      "custom_map_cleaned.pcd",
      "custom_map.pcd",
  };
  for (const auto &name : preferred) {
    fs::path candidate = p / name;
    if (fs::exists(candidate) && fs::is_regular_file(candidate)) {
      return candidate.string();
    }
  }

  // Fallback: first .pcd under the directory.
  for (const auto &entry : fs::directory_iterator(p)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() == ".pcd") {
      return entry.path().string();
    }
  }
  return {};
}

bool LoadKittiBin(const std::string &path,
                  const Options &opts,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open KITTI .bin file: " << path << "\n";
    return false;
  }
  ifs.seekg(0, std::ios::end);
  const std::streamsize num_bytes = ifs.tellg();
  if (num_bytes <= 0 || num_bytes % (sizeof(float) * 4) != 0) {
    std::cerr << "File size does not look like KITTI XYZI floats: " << path
              << "\n";
    return false;
  }
  const size_t num_points =
      static_cast<size_t>(num_bytes) / (sizeof(float) * 4);
  ifs.seekg(0, std::ios::beg);

  std::vector<float> buffer(num_points * 4);
  ifs.read(reinterpret_cast<char *>(buffer.data()),
           static_cast<std::streamsize>(buffer.size() * sizeof(float)));
  if (!ifs) {
    std::cerr << "Failed to read all bytes from " << path << "\n";
    return false;
  }

  const double max_range = opts.max_xy_range;
  cloud->points.reserve(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    const float x = buffer[4 * i];
    const float y = buffer[4 * i + 1];
    const float z = buffer[4 * i + 2];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    if (max_range > 0.0 &&
        (std::abs(x) > max_range || std::abs(y) > max_range)) {
      continue;
    }
    cloud->points.emplace_back(x, y, z);
  }
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = false;
  if (cloud->empty()) {
    std::cerr << "No valid points loaded from " << path << "\n";
    return false;
  }
  return true;
}

bool LoadPointCloud(const std::string &path,
                    const Options &opts,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  const std::string resolved = ResolvePointCloudPath(path);
  if (resolved.empty()) {
    std::cerr << "Point cloud path not found: " << path << "\n";
    return false;
  }
  const fs::path p(resolved);
  const std::string ext = p.extension().string();
  if (ext == ".pcd") {
    pcl::PCLPointCloud2 cloud_blob;
    if (pcl::io::loadPCDFile(resolved, cloud_blob) != 0) {
      std::cerr << "Failed to load PCD: " << resolved << "\n";
      return false;
    }
    pcl::fromPCLPointCloud2(cloud_blob, *cloud);
    return !cloud->empty();
  }

  // Try KITTI-style binary as fallback.
  return LoadKittiBin(resolved, opts, cloud);
}

std::string MakeOutPath(const fs::path &output_dir, const std::string &base,
                        const std::string &suffix) {
  return (output_dir / (base + "_" + suffix)).string();
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

cv::Mat ApplyColorMapWithZeroMask(const cv::Mat &normalized,
                                  const cv::Mat &zero_mask, int colormap) {
  cv::Mat colored;
  cv::applyColorMap(normalized, colored, colormap);
  if (!zero_mask.empty()) {
    colored.setTo(cv::Scalar::all(0), zero_mask);
  }
  return colored;
}

struct Grid {
  int rows{0};
  int cols{0};
  double resolution{0.2};
  avp_mapping::heatmap::Extent2D extent;
};

Grid MakeGrid(const avp_mapping::heatmap::Extent2D &extent, double resolution) {
  Grid grid;
  grid.extent = extent;
  grid.resolution = resolution;
  grid.cols = static_cast<int>(std::ceil(extent.width() / resolution));
  grid.rows = static_cast<int>(std::ceil(extent.height() / resolution));
  return grid;
}

inline bool InBounds(int row, int col, int rows, int cols) {
  return row >= 0 && row < rows && col >= 0 && col < cols;
}

struct Accumulators {
  cv::Mat counts;
  cv::Mat sum_z;
  cv::Mat sumsq_z;
  cv::Mat counts_a;
  cv::Mat counts_b;
};

Accumulators PrepareAccumulators(const Grid &grid) {
  Accumulators acc;
  const cv::Size size(grid.cols, grid.rows);
  acc.counts = cv::Mat(size, CV_32F, cv::Scalar(0));
  acc.sum_z = cv::Mat(size, CV_32F, cv::Scalar(0));
  acc.sumsq_z = cv::Mat(size, CV_32F, cv::Scalar(0));
  acc.counts_a = cv::Mat(size, CV_32F, cv::Scalar(0));
  acc.counts_b = cv::Mat(size, CV_32F, cv::Scalar(0));
  return acc;
}

void Accumulate(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
                const Grid &grid, const Options &opts, Accumulators &acc) {
  if (!cloud || cloud->empty()) {
    return;
  }
  const size_t split_index =
      static_cast<size_t>(opts.overlap_split_ratio * cloud->size());

  for (size_t i = 0; i < cloud->size(); ++i) {
    const auto &pt = cloud->at(i);
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    if (opts.max_xy_range > 0.0 &&
        (std::abs(pt.x) > opts.max_xy_range ||
         std::abs(pt.y) > opts.max_xy_range)) {
      continue;
    }
    const int col = static_cast<int>(
        std::floor((pt.x - grid.extent.min_x) / grid.resolution));
    const int row = static_cast<int>(
        std::floor((grid.extent.max_y - pt.y) / grid.resolution));
    if (!InBounds(row, col, grid.rows, grid.cols)) {
      continue;
    }
    acc.counts.at<float>(row, col) += 1.0f;
    acc.sum_z.at<float>(row, col) += static_cast<float>(pt.z);
    acc.sumsq_z.at<float>(row, col) += static_cast<float>(pt.z * pt.z);

    if (i < split_index) {
      acc.counts_a.at<float>(row, col) += 1.0f;
    } else {
      acc.counts_b.at<float>(row, col) += 1.0f;
    }
  }
}

cv::Mat ComputeOccupancyProbability(const cv::Mat &counts) {
  double max_v = 0.0;
  cv::minMaxLoc(counts, nullptr, &max_v);
  cv::Mat prob(counts.size(), CV_32F, cv::Scalar(0));
  if (max_v <= 0.0) {
    return prob;
  }
  counts.convertTo(prob, CV_32F, 1.0 / max_v, 0.0);
  return prob;
}

cv::Mat ComputeOverlap(const cv::Mat &a, const cv::Mat &b) {
  cv::Mat overlap(a.size(), CV_32F, cv::Scalar(0));
  for (int r = 0; r < a.rows; ++r) {
    const float *row_a = a.ptr<float>(r);
    const float *row_b = b.ptr<float>(r);
    float *row_o = overlap.ptr<float>(r);
    for (int c = 0; c < a.cols; ++c) {
      const float ca = row_a[c];
      const float cb = row_b[c];
      if (ca <= 0.0f || cb <= 0.0f) {
        row_o[c] = 0.0f;
        continue;
      }
      const float mn = std::min(ca, cb);
      const float mx = std::max(ca, cb);
      row_o[c] = (mx > 0.0f) ? mn / mx : 0.0f;
    }
  }
  return overlap;
}

cv::Mat ComputeNoise(const cv::Mat &counts, const cv::Mat &sum_z,
                     const cv::Mat &sumsq_z) {
  cv::Mat noise(counts.size(), CV_32F, cv::Scalar(0));
  for (int r = 0; r < counts.rows; ++r) {
    const float *row_c = counts.ptr<float>(r);
    const float *row_sum = sum_z.ptr<float>(r);
    const float *row_sq = sumsq_z.ptr<float>(r);
    float *row_n = noise.ptr<float>(r);
    for (int c = 0; c < counts.cols; ++c) {
      const float cnt = row_c[c];
      if (cnt < 2.0f) {
        row_n[c] = 0.0f;
        continue;
      }
      const float mean = row_sum[c] / cnt;
      const float mean_sq = row_sq[c] / cnt;
      const float var = std::max(0.0f, mean_sq - mean * mean);
      row_n[c] = std::sqrt(var);
    }
  }
  return noise;
}

void SaveYaml(const std::string &path, const Grid &grid,
              const std::string &label, const cv::Mat &mat) {
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    std::cerr << "Failed to open " << path << " for writing.\n";
    return;
  }
  fs << "label" << label;
  fs << "resolution_m" << grid.resolution;
  fs << "extent_min_x" << grid.extent.min_x;
  fs << "extent_max_x" << grid.extent.max_x;
  fs << "extent_min_y" << grid.extent.min_y;
  fs << "extent_max_y" << grid.extent.max_y;
  fs << "values" << mat;
  fs.release();
}

std::string BaseName(const std::string &path) {
  fs::path p(path);
  if (fs::is_directory(p)) {
    return p.filename().string();
  }
  return p.stem().string();
}

}  // namespace

int main(int argc, char **argv) {
  Options opts;
  if (!ParseArguments(argc, argv, opts)) {
    PrintUsage();
    return 1;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (!LoadPointCloud(opts.point_cloud_path, opts, cloud)) {
    return 1;
  }

  auto extent = avp_mapping::heatmap::ComputeGeometricExtent(cloud, opts.padding);
  if (!extent.isValid()) {
    std::cerr << "Failed to compute extent from point cloud.\n";
    return 1;
  }

  Grid grid = MakeGrid(extent, opts.resolution);
  if (grid.rows <= 0 || grid.cols <= 0) {
    std::cerr << "Invalid grid size computed from extent/resolution.\n";
    return 1;
  }

  fs::create_directories(opts.output_dir);
  const fs::path out_dir(opts.output_dir);
  const std::string base = BaseName(opts.point_cloud_path);

  Accumulators acc = PrepareAccumulators(grid);
  Accumulate(cloud, grid, opts, acc);

  // Occupancy / free-space confidence
  cv::Mat occupancy_prob = ComputeOccupancyProbability(acc.counts);
  cv::Mat occupancy_norm = NormalizeForHeatmap(occupancy_prob, false);
  cv::Mat occupancy_mask = occupancy_prob <= 0.0f;
  cv::Mat occupancy_vis =
      ApplyColorMapWithZeroMask(occupancy_norm, occupancy_mask,
                                cv::COLORMAP_INFERNO);
  const std::string occ_img =
      MakeOutPath(out_dir, base, "occupancy_confidence.png");
  cv::imwrite(occ_img, occupancy_vis);
  SaveYaml(MakeOutPath(out_dir, base, "occupancy_confidence.yml"), grid,
           "occupancy_probability", occupancy_prob);

  // Map consistency / overlap
  cv::Mat overlap = ComputeOverlap(acc.counts_a, acc.counts_b);
  cv::Mat overlap_norm = NormalizeForHeatmap(overlap, false);
  cv::Mat overlap_mask = overlap <= 0.0f;
  cv::Mat overlap_vis =
      ApplyColorMapWithZeroMask(overlap_norm, overlap_mask,
                                cv::COLORMAP_MAGMA);
  const std::string overlap_img =
      MakeOutPath(out_dir, base, "overlap_heatmap.png");
  cv::imwrite(overlap_img, overlap_vis);
  SaveYaml(MakeOutPath(out_dir, base, "overlap_heatmap.yml"), grid, "overlap",
           overlap);

  // Smoothness / noise
  cv::Mat noise = ComputeNoise(acc.counts, acc.sum_z, acc.sumsq_z);
  cv::Mat noise_norm = NormalizeForHeatmap(noise, true);
  cv::Mat noise_mask = noise <= 0.0f;
  cv::Mat noise_vis =
      ApplyColorMapWithZeroMask(noise_norm, noise_mask, cv::COLORMAP_PLASMA);
  const std::string noise_img =
      MakeOutPath(out_dir, base, "noise_heatmap.png");
  cv::imwrite(noise_img, noise_vis);
  SaveYaml(MakeOutPath(out_dir, base, "noise_heatmap.yml"), grid, "noise_std_z",
           noise);

  // Semantic coverage (if provided)
  if (!opts.semantic_map_path.empty()) {
    Map semantic_map;
    semantic_map.load(opts.semantic_map_path);
    auto semantic_extent =
        avp_mapping::heatmap::ComputeSemanticExtent(semantic_map, opts.padding);
    auto combined_extent =
        avp_mapping::heatmap::CombineExtents(extent, semantic_extent);
    if (combined_extent.isValid()) {
      auto semantic_heatmap =
          avp_mapping::heatmap::RenderSemanticCoverageHeatmap(
              semantic_map, combined_extent, opts.resolution);
      const std::string sem_img =
          MakeOutPath(out_dir, base, "semantic_coverage_heatmap.png");
      cv::imwrite(sem_img, semantic_heatmap.visualization);
      cv::FileStorage fs(
          MakeOutPath(out_dir, base, "semantic_coverage_heatmap.yml"),
          cv::FileStorage::WRITE);
      if (fs.isOpened()) {
        fs << "resolution_m" << semantic_heatmap.resolution;
        fs << "extent_min_x" << semantic_heatmap.extent.min_x;
        fs << "extent_max_x" << semantic_heatmap.extent.max_x;
        fs << "extent_min_y" << semantic_heatmap.extent.min_y;
        fs << "extent_max_y" << semantic_heatmap.extent.max_y;
        fs << "counts" << semantic_heatmap.counts;
        fs.release();
      }
    } else {
      std::cerr << "Semantic extent invalid; skipping semantic coverage export.\n";
    }
  } else {
    std::cout << "Semantic map not provided; semantic coverage heatmap skipped.\n";
  }

  std::cout << "Indicators saved under " << out_dir << "\n";
  return 0;
}
