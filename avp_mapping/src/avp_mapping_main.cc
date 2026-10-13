#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

#include "avp_mapping/avp_mapping.h"
#include "avp_mapping/semantic_downsampler.h"
#include "avp_mapping/slot.h"
#include "avp_mapping/heatmap_utils.h"

// draw avp map to image
cv::Mat drawMap(const Map &avp_map, const std::vector<Slot> &cur_slots = {}){
  // compute map size
  auto box = avp_map.getBoundingBox();
  if (box.isEmpty()){ // empty map
    box.extend(Eigen::Vector3d(0, 0, 0));
  }
  Eigen::Vector3d bl = box.min() + Eigen::Vector3d(-1, -1, -1);
  Eigen::Vector3d tr = box.max() + Eigen::Vector3d(1, 1, 1);
  int width = (tr.x() - bl.x()) * kPixelScaleInv;
  int height = (tr.y() - bl.y()) * kPixelScaleInv;
  cv::Mat map(height, width, CV_8UC1, 91);
  auto to_pixel = [&](const Eigen::Vector3d &pt) {
    return cv::Point2f((pt.x() - bl.x()) * kPixelScaleInv, height - (pt.y() - bl.y()) * kPixelScaleInv);
  };
  // draw semantic elements
  auto grid_offset = to_pixel(Eigen::Vector3d(0, 0, 0));
  const auto dash_line = avp_map.getSemanticElement(SemanticLabel::kDashLine);
  for (const auto &pt : dash_line) {
    cv::circle(map, grid_offset + cv::Point2f(pt.x(), -pt.y()), 1, kDashGray, 1);
  }
  const auto arrow_line = avp_map.getSemanticElement(SemanticLabel::kArrowLine);
  for (const auto &pt : arrow_line) {
    cv::circle(map, grid_offset + cv::Point2f(pt.x(), -pt.y()), 1, kArrowGray, 1);
  }
  // draw slots
  auto draw_slot = [&](const std::vector<Slot> &slots, uchar color){
    std::vector<cv::Point2f> slot_corners(4);
    for (const auto &slot : slots) {
      for (int i = 0; i < 4; ++i) {
        slot_corners[i] = to_pixel(slot.corners_[i]);
      }
      cv::line(map, slot_corners[0], slot_corners[1], color, 2);
      cv::line(map, slot_corners[1], slot_corners[2], color, 2);
      cv::line(map, slot_corners[2], slot_corners[3], color, 2);
      cv::line(map, slot_corners[3], slot_corners[0], color, 2);
    }
  };
  draw_slot(avp_map.getAllSlots(), kSlotGray); //slots in map
  draw_slot(cur_slots, 0); //slots in current frame
  return map;
}

// opencv viewer
class CViewer : public ViewerInterface{
 public:
  CViewer(bool imshow, bool autorun) : ViewerInterface(imshow, autorun) {
    if (imshow_){
      cv::namedWindow("map", cv::WINDOW_NORMAL);
    }
  }
  ~CViewer() = default;
  void displayAvpMap(const TimedPose &pose, const Map &avp_map, const std::vector<Slot> &cur_slots) override {
    if (imshow_) {
      auto map = drawMap(avp_map, cur_slots);
      cv::imshow("map", map);
    }
    std::cout << "\tmap have " << avp_map.getAllSlots().size() << " slots" << std::endl;
  }

};

int main(int argc, char **argv) {
  std::string path(DATASET_PATH); // default dataset path
  if (argc > 1) {
    path = std::string(argv[1]);  // user specified dataset path
  }

  if (const char *voxel_env = std::getenv("AVP_SEMANTIC_VOXEL")) {
    double voxel_size = std::atof(voxel_env);
    if (voxel_size > 0.0) {
      avp_mapping::SemanticDownsampler::Config cfg;
      cfg.voxel_size = std::max(voxel_size, kPixelScale);
      avp_mapping::SetSemanticDownsamplerConfig(cfg);
      std::cout << "Semantic downsampling voxel size set to "
                << cfg.voxel_size << " m via AVP_SEMANTIC_VOXEL" << std::endl;
    }
  }
  bool imshow {true}, autorun {true};
  auto viewer = std::make_shared<CViewer>(imshow, autorun);
  AvpMapping mapping;
  mapping.setViewer(viewer);

  // load pose data
  std::string pose_file(path + "/poses.bin");
  std::ifstream pose_ifs(pose_file, std::ios::binary);
  if (!pose_ifs) {
    std::cerr << "Unable to open pose file: " << pose_file << std::endl;
    return 0;
  }
  // load pose data
  std::vector<TimedPose> poses;
  io::loadVector(pose_ifs, poses);
  for (const auto &pose : poses) {
    mapping.processPose(pose);
  }
  std::cout << "poses num: " << poses.size() << std::endl;
  pose_ifs.close();

  // load and process images
  std::ifstream ipm_imgs_ifs(path + "/ipm_imgs.txt");
  if (!ipm_imgs_ifs) {
    std::cerr << "Unable to open ipm img file: " << path + "/ipm_imgs.txt" << std::endl;
    return 0;
  }
  double time;
  std::string line, img_name;
  while (std::getline(ipm_imgs_ifs, line)) {
    std::stringstream ss(line);
    ss >> time >> img_name;
    std::cout << std::to_string(time) << " :  " << img_name << std::endl;
    // process image
    mapping.processImage(time, cv::imread(path + "/" + img_name));
  }

  // save map data
  mapping.getMap().save(path + "/avp_map.bin");
  cv::imwrite(path + "/avp_map.png", drawMap(mapping.getMap()));
  auto semantic_extent =
      avp_mapping::heatmap::ComputeSemanticExtent(mapping.getMap(), 1.0);
  if (semantic_extent.isValid()) {
    auto heatmap =
        avp_mapping::heatmap::RenderSemanticCoverageHeatmap(mapping.getMap(),
                                                            semantic_extent,
                                                            0.5);
    const std::string heatmap_path = path + "/semantic_coverage_heatmap.png";
    cv::imwrite(heatmap_path, heatmap.visualization);
    cv::FileStorage fs(heatmap_path + ".yml", cv::FileStorage::WRITE);
    fs << "resolution_m" << heatmap.resolution;
    fs << "extent_min_x" << heatmap.extent.min_x;
    fs << "extent_max_x" << heatmap.extent.max_x;
    fs << "extent_min_y" << heatmap.extent.min_y;
    fs << "extent_max_y" << heatmap.extent.max_y;
    fs << "counts" << heatmap.counts;
    fs.release();
    std::cout << "semantic coverage heatmap saved to " << heatmap_path << std::endl;
  }
  std::cout << "map image saved to " << path + "/avp_map.png" << std::endl;
  return 0;
}
