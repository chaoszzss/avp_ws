#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

#include "../include/avp_mapping/avp_mapping.h"

namespace {
constexpr const char kSlotOutputDir[] =
    "/home/chaoszzss/avp_ws/src/avp_mapping/src/AVPMapping/output/"
    "Quantitative_indicators/slot_det1";
constexpr const char kSlotMetricsDir[] =
    "/home/chaoszzss/avp_ws/src/avp_mapping/src/AVPMapping/output/"
    "Quantitative_indicators/slot_det2";

bool slotDumpEnabled() {
  static bool enabled = []() {
    const char *flag = std::getenv("AVP_SLOT_DUMP");
    return flag && std::strcmp(flag, "0") != 0;
  }();
  return enabled;
}

// 车位面积的比例阈值。交叠像素超过 车位面积 × 8% 时才算占用；数值越大越严格
constexpr double kVehicleOverlapRatioThreshold = 0.08; // 0.08
// 车位与车辆掩膜相交的像素数下限，即使车位面积很小也至少要达到这个像素数才判定为占用，
// 防止偶然落入的几百个噪声点触发占用
constexpr int kVehicleOverlapMinPixels = 2000;  // 1500 -> 2000
// 连通域面积下限。小于这个像素数的车辆掩膜连通域会被滤掉，用于清除孤立噪点和极小的碎片
constexpr int kVehicleMaskMinComponentPx = 1200; // 1200

struct FrameMetricsRow {
  size_t frame_id = 0;
  double stamp = 0.0;
  size_t slots_detected = 0;
  size_t slots_occupied = 0;
  size_t slots_new = 0;
  double avg_energy = 0.0;
  double median_energy = 0.0;
  double min_visibility = 0.0;
  double nms_ratio = 0.0;
  size_t num_hypotheses = 0;
  size_t num_after_nms = 0;
};

void appendFrameMetrics(const FrameMetricsRow &row) {
  static std::mutex metrics_mutex;
  std::lock_guard<std::mutex> lock(metrics_mutex);
  static std::ofstream ofs;
  if (!ofs.is_open()) {
    std::filesystem::create_directories(kSlotMetricsDir);
    const auto metrics_path =
        std::filesystem::path(kSlotMetricsDir) / "frame_metrics.csv";
    ofs.open(metrics_path, std::ios::app);
    if (!ofs) {
      return;
    }
    if (ofs.tellp() == 0) {
      ofs << "frame_id,time_stamp,slots_detected,slots_occupied,slots_new,"
             "avg_energy,median_energy,min_visibility,nms_ratio,"
             "num_hypotheses,num_after_nms\n";
    }
  }
  ofs << row.frame_id << ','
      << std::fixed << std::setprecision(6) << row.stamp << ','
      << row.slots_detected << ',' << row.slots_occupied << ','
      << row.slots_new << ','
      << std::setprecision(3) << row.avg_energy << ','
      << row.median_energy << ',' << row.min_visibility << ','
      << std::setprecision(4) << row.nms_ratio << ','
      << row.num_hypotheses << ',' << row.num_after_nms << '\n';
  ofs.flush();
}

void dumpDebugImage(const std::string &tag, const cv::Mat &image) {
  if (!slotDumpEnabled()) {
    return;
  }
  static std::mutex dump_mutex;
  static size_t dump_index = 0;
  std::lock_guard<std::mutex> lock(dump_mutex);
  std::filesystem::create_directories(kSlotOutputDir);
  std::ostringstream oss;
  oss << kSlotOutputDir << "/debug_"
      << std::setw(5) << std::setfill('0') << dump_index++ << "_" << tag << ".png";
  cv::Mat to_write;
  if (image.type() == CV_8UC1) {
    cv::cvtColor(image, to_write, cv::COLOR_GRAY2BGR);
  } else {
    to_write = image.clone();
  }
  cv::imwrite(oss.str(), to_write);
}

void filterSmallComponents(cv::Mat &binary_mask, int min_pixels) {
  if (binary_mask.empty()) {
    return;
  }
  CV_Assert(binary_mask.type() == CV_8UC1);
  cv::Mat labels, stats, centroids;
  const int num_components =
      cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);

  if (num_components <= 1) {
    return;
  }

  cv::Mat filtered = cv::Mat::zeros(binary_mask.size(), binary_mask.type());
  for (int i = 1; i < num_components; ++i) {
    const int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area >= min_pixels) {
      filtered.setTo(255, labels == i);
    }
  }
  binary_mask = filtered;
}
}  // namespace

AvpMapping::AvpMapping() {
  pre_key_pose_.time_ = -1;
  T_vehicle_ipm_.linear().setIdentity();
  T_vehicle_ipm_.translation() = Eigen::Vector3d(0.0, 1.32, 0.0);
}

bool AvpMapping::isKeyFrame(const TimedPose &pose) {
  if ((pose.t_ - pre_key_pose_.t_).norm() > 0.1 ||
      std::fabs(GetYaw(pre_key_pose_.R_.inverse() * pose.R_)) > 5. * kToRad ||
      pose.time_ - pre_key_pose_.time_ > 30 || pre_key_pose_.time_ < -1) {
    pre_key_pose_ = pose;
    return true;
  }
  return false;
}

void AvpMapping::processPose(const TimedPose &pose) {
  pose_interpolation_.Push(pose);
}

void AvpMapping::processImage(double time, const cv::Mat &ipm_seg_img) {
  if (time < pose_interpolation_.EarliestTime()) {
    return;
  }
  TimedPose pose;
  pose.time_ = time;
  if (pose_interpolation_.LookUp(pose) && isKeyFrame(pose)) {
    pose_interpolation_.TrimBefore(time);
    std::cout << "keyframe : " << "  t = " << pose.t_.transpose().head(2)
              << ", yaw = " << GetYaw(pose.R_) * kToDeg << std::endl;

    cv::Mat img_gray;
    cv::cvtColor(ipm_seg_img, img_gray, cv::COLOR_BGR2GRAY);
    extractSlot(img_gray, pose);  // 只传递灰度图像
  }
}

/// 将IPM图像坐标系转换为世界坐标系
Eigen::Vector3d AvpMapping::ipmPlane2Global(const TimedPose &T_world_vehicle, const cv::Point2f &ipm_point){
  Eigen::Vector3d pt_global;
  //////////////////////// TODO: transform ipm pixel to point in global ///////////////////////
  // 从ipm图像坐标系 --> 车辆中心坐标系
  Eigen::Vector3d pt_vehicle_center;
  pt_vehicle_center.x() = -(500 - ipm_point.x) * 0.02;
  pt_vehicle_center.y() = (500 - ipm_point.y) * 0.02;
  pt_vehicle_center.z() = 0;

  // 车辆中心坐标系 --> （车辆坐标系）车辆后轮连杆中点
  Eigen::Vector3d pt_vehicle;
  pt_vehicle.x() = pt_vehicle_center.x();
  pt_vehicle.y() = pt_vehicle_center.y() + 1.32;
  pt_vehicle.z() = pt_vehicle_center.z();


  // 车辆坐标系 --> 世界坐标系 (已给出作为先验信息：如需计算可使用ICP进行匹配)
  pt_global = T_world_vehicle.R_ * pt_vehicle + T_world_vehicle.t_;

  return pt_global;
}

void AvpMapping::extractSlot(const cv::Mat &img_gray, const TimedPose &T_world_vehicle) {
  cv::Mat slot_img = cv::Mat::zeros(img_gray.size(), img_gray.type());
  for (int i = 0; i < img_gray.rows; ++i) {
    for (int j = 0; j < img_gray.cols; ++j) {
      const uchar pixel = img_gray.at<uchar>(i, j);
      if (kSlotGray == pixel || kSlotGray1 == pixel) {
        // avp_map_.addSemanticElement(SemanticLabel::kSlot,
        //                             {ipmPlane2Global(T_world_vehicle, cv::Point2f(j, i))});
        slot_img.at<uchar>(i, j) = 254;
      } else if(kArrowGray == img_gray.at<uchar>(i, j)) {
        avp_map_.addSemanticElement(SemanticLabel::kArrowLine,
                                    {ipmPlane2Global(T_world_vehicle, cv::Point2f(j, i))});
      } else if(kDashGray == img_gray.at<uchar>(i, j)) {
        avp_map_.addSemanticElement(SemanticLabel::kDashLine,
                                    {ipmPlane2Global(T_world_vehicle, cv::Point2f(j, i))});
      } else{}
    }
  }
  // dumpDebugImage("slot_raw", slot_img);
  cv::Mat vehicle_mask;
  cv::inRange(img_gray, 0, 20, vehicle_mask);
  cv::Mat vehicle_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                     cv::Size(9, 9));
  cv::morphologyEx(vehicle_mask, vehicle_mask, cv::MORPH_CLOSE, vehicle_kernel);
  cv::Mat vehicle_open_kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(vehicle_mask, vehicle_mask, cv::MORPH_OPEN, vehicle_open_kernel);
  filterSmallComponents(vehicle_mask, kVehicleMaskMinComponentPx);

  int kernelSize = 13;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                             cv::Size(kernelSize, kernelSize));
  cv::Mat closed;
  cv::morphologyEx(slot_img, closed, cv::MORPH_CLOSE, kernel);
  // dumpDebugImage("slot_closed", closed);
  auto line_img = closed.clone();

  cv::Mat skel = skeletonize(closed);
  // dumpDebugImage("slot_skeleton", skel);
  removeIsolatedPixels(skel, 1);
  // extract slot lines
  std::vector<cv::Vec4i> lines;
  // cv::HoughLinesP(skel, lines, 1, CV_PI / 180, 50, 50, 50);
  cv::HoughLinesP(skel, lines, 1, CV_PI / 180, 60, 80, 30);
  // detect slots
  auto [corners, slot_points, slot_energies, slot_visibilities, detection_stats] =
      detectSlot(slot_img, vehicle_mask, lines);

  std::vector<bool> occupancy_status;
  occupancy_status.reserve(slot_points.size() / 4);
  std::vector<int> overlap_pixels_list;
  std::vector<int> slot_area_list;
  std::vector<cv::Point2f> slot_centers;
  std::vector<std::array<cv::Point2f, 4>> slot_polygons;
  cv::Mat overlap_accum = cv::Mat::zeros(img_gray.size(), CV_8UC1);
  for (int i = 0; i < slot_points.size(); i += 4) {
    bool occupied = false;
    
    std::vector<cv::Point2f> slot_corners = {
      slot_points[i], slot_points[i+1], slot_points[i+2], slot_points[i+3]
    };
    slot_polygons.push_back(std::array<cv::Point2f, 4>{
        slot_corners[0], slot_corners[1], slot_corners[2], slot_corners[3]});
    
    cv::Mat mask = cv::Mat::zeros(img_gray.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> polygons = {{
      cv::Point(slot_corners[0].x, slot_corners[0].y),
      cv::Point(slot_corners[1].x, slot_corners[1].y),
      cv::Point(slot_corners[2].x, slot_corners[2].y),
      cv::Point(slot_corners[3].x, slot_corners[3].y)
    }};
    cv::fillPoly(mask, polygons, cv::Scalar(255));
    
    cv::Mat overlap;
    cv::bitwise_and(vehicle_mask, mask, overlap);
    cv::bitwise_or(overlap_accum, overlap, overlap_accum);
    const int overlap_pixels = cv::countNonZero(overlap);
    const int slot_area_pixels = std::max(1, cv::countNonZero(mask));
    const int ratio_threshold =
        static_cast<int>(std::round(slot_area_pixels * kVehicleOverlapRatioThreshold));
    const int min_required = std::max(kVehicleOverlapMinPixels, ratio_threshold);
    occupied = overlap_pixels >= min_required;
    
    overlap_pixels_list.push_back(overlap_pixels);
    slot_area_list.push_back(slot_area_pixels);
    const cv::Point2f center = (slot_corners[0] + slot_corners[1] +
                                slot_corners[2] + slot_corners[3]) * 0.25f;
    slot_centers.push_back(center);

    occupancy_status.push_back(occupied);
  }

  // dumpDebugImage("vehicle_mask", vehicle_mask);
  // dumpDebugImage("vehicle_overlap_mask", overlap_accum);

  if (!slot_polygons.empty()) {
    cv::Mat overlay_vis;
    cv::cvtColor(img_gray, overlay_vis, cv::COLOR_GRAY2BGR);

    cv::Mat vehicle_color_overlay(overlay_vis.size(), CV_8UC3, cv::Scalar(255, 0, 0));
    vehicle_color_overlay.copyTo(overlay_vis, vehicle_mask);
    cv::Mat overlap_color_overlay(overlay_vis.size(), CV_8UC3, cv::Scalar(0, 0, 255));
    overlap_color_overlay.copyTo(overlay_vis, overlap_accum);

    for (size_t idx = 0; idx < slot_polygons.size(); ++idx) {
      const auto &poly = slot_polygons[idx];
      std::vector<cv::Point> contour;
      contour.reserve(4);
      for (const auto &pt : poly) {
        contour.emplace_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
      }
      const cv::Scalar slot_color = occupancy_status[idx]
                                        ? cv::Scalar(0, 0, 255)
                                        : cv::Scalar(0, 255, 0);
      const std::vector<std::vector<cv::Point>> contours = {contour};
      cv::polylines(overlay_vis, contours, true, slot_color, 2);

      const cv::Point center_pt(std::clamp(static_cast<int>(std::round(slot_centers[idx].x)),
                                           0, overlay_vis.cols - 1),
                                std::clamp(static_cast<int>(std::round(slot_centers[idx].y)),
                                           0, overlay_vis.rows - 1));
      const double ratio =
          static_cast<double>(overlap_pixels_list[idx]) /
          std::max(1, slot_area_list[idx]);
      const std::string info = cv::format("ov:%d area:%d r:%.2f %s",
                                          overlap_pixels_list[idx],
                                          slot_area_list[idx],
                                          ratio,
                                          occupancy_status[idx] ? "X" : "O");
      cv::Point text_pt = center_pt + cv::Point(-60, -10);
      text_pt.x = std::clamp(text_pt.x, 0, std::max(0, overlay_vis.cols - 100));
      text_pt.y = std::clamp(text_pt.y, 0, std::max(0, overlay_vis.rows - 10));
      cv::putText(overlay_vis, info, text_pt, cv::FONT_HERSHEY_PLAIN, 1.0,
                  cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
    // dumpDebugImage("slot_overlap_overlay", overlay_vis);
  }

  // add detected slots to map
  std::vector<Slot> slots(slot_points.size() / 4);
  int slot_index = 0;  // 使用单独的计数器
  size_t new_slots_count = 0;
  for (auto &slot : slots){
    slot.corners_[0] = ipmPlane2Global(T_world_vehicle, slot_points[4*slot_index]);
    slot.corners_[1] = ipmPlane2Global(T_world_vehicle, slot_points[4*slot_index+1]);
    slot.corners_[2] = ipmPlane2Global(T_world_vehicle, slot_points[4*slot_index+2]);
    slot.corners_[3] = ipmPlane2Global(T_world_vehicle, slot_points[4*slot_index+3]);
    const bool observed = occupancy_status[slot_index];
    bool is_new_slot = false;
    size_t map_index = avp_map_.addSlot(slot, &is_new_slot);
    if (is_new_slot) {
      ++new_slots_count;
    }
    const bool filtered = avp_map_.updateSlotOccupancy(map_index, observed);
    slot.occupied = filtered;
    slot.occupancy_score = avp_map_.getSlotScore(map_index);
    occupancy_status[slot_index] = filtered;
    slot_index++;
  }
  
  // update viewer
  if (viewer_){
    viewer_->displayAvpMap(T_world_vehicle, avp_map_, slots);
    viewer_->displayIpmDetection(corners, slot_points, img_gray,
                                 occupancy_status, slot_energies, slot_visibilities);
  }

  FrameMetricsRow metrics;
  metrics.frame_id = detection_stats.frame_id;
  metrics.stamp = T_world_vehicle.time_;
  metrics.slots_detected = slots.size();
  metrics.slots_occupied =
      std::count(occupancy_status.begin(), occupancy_status.end(), true);
  metrics.slots_new = new_slots_count;
  metrics.num_hypotheses = detection_stats.num_hypotheses;
  metrics.num_after_nms = detection_stats.num_after_nms;
  metrics.nms_ratio =
      detection_stats.num_hypotheses == 0
          ? 0.0
          : static_cast<double>(detection_stats.num_after_nms) /
                static_cast<double>(detection_stats.num_hypotheses);
  if (!slot_energies.empty()) {
    const double sum_energy =
        std::accumulate(slot_energies.begin(), slot_energies.end(), 0.0);
    metrics.avg_energy = sum_energy / static_cast<double>(slot_energies.size());
    std::vector<float> energies_copy = slot_energies;
    const size_t mid = energies_copy.size() / 2;
    std::nth_element(energies_copy.begin(),
                     energies_copy.begin() + mid,
                     energies_copy.end());
    if (energies_copy.size() % 2 == 0) {
      const float upper = energies_copy[mid];
      std::nth_element(energies_copy.begin(),
                       energies_copy.begin() + mid - 1,
                       energies_copy.end());
      const float lower = energies_copy[mid - 1];
      metrics.median_energy = 0.5 * (upper + lower);
    } else {
      metrics.median_energy = energies_copy[mid];
    }
  }
  if (!slot_visibilities.empty()) {
    double min_vis = std::numeric_limits<double>::infinity();
    for (const auto &vis : slot_visibilities) {
      for (float value : vis) {
        if (value < min_vis) {
          min_vis = value;
        }
      }
    }
    metrics.min_visibility = std::isfinite(min_vis) ? min_vis : 0.0;
  }
  appendFrameMetrics(metrics);
}
