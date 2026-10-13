#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <pcl/io/pcd_io.h>

#include "avp_mapping/heatmap_utils.h"
#include "avp_mapping/map.h"
#include "avp_mapping/slot.h"

namespace {

struct Options {
  std::string semantic_map_path;
  std::string geometric_map_path;
  std::string output_path{
      "/home/chaoszzss/avp_ws/src/avp_mapping/src/AVPMapping/output/"
      "topology_overlay.svg"};
  double resolution{0.2};  // meters per pixel for rasterization
  double padding{1.0};
  int geometry_stride{4};  // subsample background points
};

struct PixelNode {
  int id{-1};
  cv::Point coord;
  Eigen::Vector2d world;
};

struct PixelEdge {
  int from{-1};
  int to{-1};
  std::vector<Eigen::Vector2d> polyline;
  double length{0.0};
};

using NeighborOffsets = std::array<cv::Point, 8>;

const NeighborOffsets &GetNeighbors() {
  static NeighborOffsets offsets{{
      cv::Point(-1, -1), cv::Point(0, -1),  cv::Point(1, -1),
      cv::Point(-1, 0),  cv::Point(1, 0),   cv::Point(-1, 1),
      cv::Point(0, 1),   cv::Point(1, 1),
  }};
  return offsets;
}

Options ParseArguments(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto next = [&](double *target) {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + arg);
      }
      *target = std::stod(argv[++i]);
    };
    if (arg == "--semantic_map") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --semantic_map");
      }
      opts.semantic_map_path = argv[++i];
    } else if (arg == "--geometric_map") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --geometric_map");
      }
      opts.geometric_map_path = argv[++i];
    } else if (arg == "--output") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --output");
      }
      opts.output_path = argv[++i];
    } else if (arg == "--resolution") {
      next(&opts.resolution);
    } else if (arg == "--padding") {
      next(&opts.padding);
    } else if (arg == "--stride") {
      double temp = 0.0;
      next(&temp);
      opts.geometry_stride = static_cast<int>(std::max(1.0, temp));
    } else if (arg == "--help") {
      std::cout << "Usage: topology_svg_tool --semantic_map path "
                   "--geometric_map path [--output path.svg] "
                   "[--resolution meters] [--padding meters] [--stride N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (opts.semantic_map_path.empty() || opts.geometric_map_path.empty()) {
    throw std::runtime_error("semantic_map and geometric_map paths are required");
  }
  return opts;
}

cv::Mat RasterizeSemantic(const Map &map,
                          const avp_mapping::heatmap::Extent2D &extent,
                          double resolution_m) {
  const int cols =
      static_cast<int>(std::ceil(extent.width() / resolution_m));
  const int rows =
      static_cast<int>(std::ceil(extent.height() / resolution_m));
  cv::Mat raster(rows, cols, CV_8UC1, cv::Scalar(0));

  auto accumulate = [&](double x, double y) {
    const int col =
        static_cast<int>(std::floor((x - extent.min_x) / resolution_m));
    const int row =
        static_cast<int>(std::floor((extent.max_y - y) / resolution_m));
    if (col < 0 || col >= cols || row < 0 || row >= rows) {
      return;
    }
    raster.at<uint8_t>(row, col) = 255;
  };

  const auto dash = map.getSemanticElement(SemanticLabel::kDashLine);
  for (const auto &pt : dash) {
    accumulate(pt.x() * kPixelScale, pt.y() * kPixelScale);
  }
  const auto arrow = map.getSemanticElement(SemanticLabel::kArrowLine);
  for (const auto &pt : arrow) {
    accumulate(pt.x() * kPixelScale, pt.y() * kPixelScale);
  }
  const auto slots = map.getAllSlots();
  for (const auto &slot : slots) {
    for (const auto &corner : slot.corners_) {
      accumulate(corner.x(), corner.y());
    }
  }

  cv::Mat dilated;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
  cv::dilate(raster, dilated, kernel);
  return dilated;
}

Eigen::Vector2d PixelToWorld(const cv::Point &px,
                             const avp_mapping::heatmap::Extent2D &extent,
                             double resolution_m) {
  const double x = extent.min_x + (px.x + 0.5) * resolution_m;
  const double y = extent.max_y - (px.y + 0.5) * resolution_m;
  return {x, y};
}

std::vector<cv::Point> CollectNeighbors(const cv::Mat &skeleton,
                                        const cv::Point &pt) {
  std::vector<cv::Point> neighbors;
  for (const auto &off : GetNeighbors()) {
    const cv::Point next = pt + off;
    if (next.x < 0 || next.x >= skeleton.cols ||
        next.y < 0 || next.y >= skeleton.rows) {
      continue;
    }
    if (skeleton.at<uint8_t>(next) > 0) {
      neighbors.push_back(next);
    }
  }
  return neighbors;
}

struct GraphExtractionResult {
  std::vector<PixelNode> nodes;
  std::vector<PixelEdge> edges;
};

GraphExtractionResult BuildGraphFromSkeleton(
    const cv::Mat &skeleton, const avp_mapping::heatmap::Extent2D &extent,
    double resolution_m) {
  GraphExtractionResult graph;
  if (skeleton.empty()) {
    return graph;
  }

  const int rows = skeleton.rows;
  const int cols = skeleton.cols;
  std::vector<int> pixel_to_node(rows * cols, -1);
  auto idx = [&](const cv::Point &pt) { return pt.y * cols + pt.x; };

  auto make_node = [&](const cv::Point &pt) -> int {
    const int id = static_cast<int>(graph.nodes.size());
    graph.nodes.push_back(
        PixelNode{id, pt, PixelToWorld(pt, extent, resolution_m)});
    pixel_to_node[idx(pt)] = id;
    return id;
  };

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (skeleton.at<uint8_t>(r, c) == 0) {
        continue;
      }
      const cv::Point pt(c, r);
      const auto neighbors = CollectNeighbors(skeleton, pt);
      if (neighbors.size() != 2) {
        make_node(pt);
      }
    }
  }

  auto ensure_node = [&](const cv::Point &pt) -> int {
    const int id = pixel_to_node[idx(pt)];
    if (id >= 0) {
      return id;
    }
    return make_node(pt);
  };

  cv::Mat visited = cv::Mat::zeros(rows, cols, CV_8UC1);

  auto mark_edge = [&](const cv::Point &a, const cv::Point &b) {
    visited.at<uint8_t>(a) = 255;
    visited.at<uint8_t>(b) = 255;
  };

  auto trace = [&](const cv::Point &start, const cv::Point &next) {
    const int from_id = ensure_node(start);
    cv::Point prev = start;
    cv::Point current = next;
    std::vector<cv::Point> polyline_px{start, next};

    while (true) {
      mark_edge(prev, current);
      auto neighbors = CollectNeighbors(skeleton, current);
      neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), prev),
                      neighbors.end());
      if (neighbors.empty()) {
        const int to_id = ensure_node(current);
        if (to_id != from_id) {
          PixelEdge edge;
          edge.from = from_id;
          edge.to = to_id;
          for (const auto &px : polyline_px) {
            Eigen::Vector2d world = PixelToWorld(px, extent, resolution_m);
            edge.polyline.push_back(world);
          }
          for (size_t i = 1; i < edge.polyline.size(); ++i) {
            edge.length += (edge.polyline[i] - edge.polyline[i - 1]).norm();
          }
          graph.edges.push_back(edge);
        }
        return;
      }
      if (neighbors.size() > 1) {
        const int to_id = ensure_node(current);
        if (to_id != from_id) {
          PixelEdge edge;
          edge.from = from_id;
          edge.to = to_id;
          for (const auto &px : polyline_px) {
            Eigen::Vector2d world = PixelToWorld(px, extent, resolution_m);
            edge.polyline.push_back(world);
          }
          for (size_t i = 1; i < edge.polyline.size(); ++i) {
            edge.length += (edge.polyline[i] - edge.polyline[i - 1]).norm();
          }
          graph.edges.push_back(edge);
        }
        return;
      }
      prev = current;
      current = neighbors.front();
      polyline_px.push_back(current);
      if (visited.at<uint8_t>(current) > 0) {
        return;
      }
    }
  };

  for (const auto &node : graph.nodes) {
    const auto neighbors =
        CollectNeighbors(skeleton, node.coord);
    for (const auto &next : neighbors) {
      if (visited.at<uint8_t>(next) == 0) {
        trace(node.coord, next);
      }
    }
  }

  return graph;
}

void EnsureDirectory(const std::string &path) {
  const auto slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return;
  }
  const std::string dir = path.substr(0, slash);
  if (dir.empty()) {
    return;
  }
  const std::string command = "mkdir -p \"" + dir + "\"";
  (void)std::system(command.c_str());
}

cv::Point2d WorldToSvg(const Eigen::Vector2d &world,
                       const avp_mapping::heatmap::Extent2D &extent,
                       double width, double height) {
  const double nx = (world.x() - extent.min_x) / extent.width();
  const double ny = (extent.max_y - world.y()) / extent.height();
  return {nx * width, ny * height};
}

void ExportSvg(const GraphExtractionResult &graph,
               const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
               const avp_mapping::heatmap::Extent2D &extent,
               const Options &options) {
  if (!extent.isValid() || graph.nodes.empty()) {
    std::cerr << "Nothing to export.\n";
    return;
  }

  const double aspect = extent.height() / extent.width();
  const double width = 1200.0;
  const double height = width * aspect;

  EnsureDirectory(options.output_path);
  std::ofstream ofs(options.output_path);
  if (!ofs.is_open()) {
    throw std::runtime_error("Unable to open output file: " +
                             options.output_path);
  }

  ofs << "<svg xmlns='http://www.w3.org/2000/svg' "
         "width='"
      << width << "' height='" << height << "' viewBox='0 0 " << width << " "
      << height << "'>\n";
  ofs << "<rect width='100%' height='100%' fill='#050607'/>\n";

  ofs << "<defs>\n";
  ofs <<
      "<marker id='arrowhead' markerWidth='6' markerHeight='6' "
      "refX='5' refY='3' orient='auto' markerUnits='strokeWidth'>"
      "<path d='M0,0 L6,3 L0,6 z' fill='#FFB347'/></marker>\n";
  ofs << "</defs>\n";

  if (cloud && !cloud->empty()) {
    const int stride = std::max(1, options.geometry_stride);
    ofs << "<g fill='none' stroke='#49C2A2' stroke-opacity='0.25'>\n";
    for (size_t i = 0; i < cloud->size(); i += stride) {
      const auto &pt = cloud->points[i];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) {
        continue;
      }
      Eigen::Vector2d world(pt.x, pt.y);
      auto svg = WorldToSvg(world, extent, width, height);
      ofs << "<circle cx='" << svg.x << "' cy='" << svg.y
          << "' r='0.8'/>\n";
    }
    ofs << "</g>\n";
  }

  ofs << "<g fill='none' stroke='#FFB347' stroke-width='2' "
         "stroke-linecap='round' stroke-linejoin='round' "
         "marker-end='url(#arrowhead)'>\n";
  for (const auto &edge : graph.edges) {
    if (edge.polyline.size() < 2) {
      continue;
    }
    ofs << "<polyline points='";
    for (size_t i = 0; i < edge.polyline.size(); ++i) {
      auto svg = WorldToSvg(edge.polyline[i], extent, width, height);
      ofs << svg.x << "," << svg.y;
      if (i + 1 != edge.polyline.size()) {
        ofs << " ";
      }
    }
    ofs << "' />\n";
  }
  ofs << "</g>\n";

  ofs << "<g fill='#D4F1F9' stroke='#0D1F2D' stroke-width='0.5'>\n";
  for (const auto &node : graph.nodes) {
    auto svg = WorldToSvg(node.world, extent, width, height);
    ofs << "<circle cx='" << svg.x << "' cy='" << svg.y
        << "' r='3'/>\n";
  }
  ofs << "</g>\n";

  ofs << "</svg>\n";
  ofs.close();
  std::cout << "Topology SVG exported to " << options.output_path << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = ParseArguments(argc, argv);

    Map semantic_map;
    semantic_map.load(options.semantic_map_path);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(options.geometric_map_path, *cloud) != 0) {
      std::cerr << "Failed to read geometric map: "
                << options.geometric_map_path << "\n";
      return 1;
    }

    auto semantic_extent =
        avp_mapping::heatmap::ComputeSemanticExtent(semantic_map,
                                                    options.padding);
    auto geometric_extent =
        avp_mapping::heatmap::ComputeGeometricExtent(cloud,
                                                     options.padding);
    auto extent =
        avp_mapping::heatmap::CombineExtents(semantic_extent,
                                             geometric_extent);
    if (!extent.isValid()) {
      std::cerr << "Unable to derive extent from inputs.\n";
      return 1;
    }

    const cv::Mat semantic_raster =
        RasterizeSemantic(semantic_map, extent, options.resolution);
    cv::Mat skeleton = skeletonize(semantic_raster);
    auto graph = BuildGraphFromSkeleton(skeleton, extent,
                                        options.resolution);

    ExportSvg(graph, cloud, extent, options);
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 1;
  }
  return 0;
}

