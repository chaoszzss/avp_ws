#pragma once
#include <vector>
#include <memory>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "avp_mapping/pose_interpolation.h"
#include "avp_mapping/map.h"
#include "avp_mapping/viewer_interface.h"

class AvpMapping {
 public:
  AvpMapping();
  ~AvpMapping() = default;
  void processPose(const TimedPose &pose);
  void processImage(double time, const cv::Mat &ipm_seg_img);

  const Map &getMap() const { return avp_map_; }
  void setViewer(std::shared_ptr<ViewerInterface> viewer) { viewer_ = viewer; }

 private:
  bool isKeyFrame(const TimedPose &pose);
  Eigen::Vector3d ipmPlane2Global(const TimedPose &T_world_vehicle, const cv::Point2f &ipm_point);
  void extractSlot(const cv::Mat &slot_img, const TimedPose &T_world_vehicle);

  Eigen::Affine3d T_vehicle_ipm_;
  TimedPose pre_key_pose_; // Pose of last key frame
  Map avp_map_; // avp map data

  PoseInterpolation pose_interpolation_; // T_world_vehicle
  std::shared_ptr<ViewerInterface> viewer_{nullptr}; // viewer for debug
};
