#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <ros/package.h>

#include "../include/avp_localization.h"  // 
#include <pcl/common/transforms.h>        // 用于处理点云的坐标变换

namespace {
constexpr std::size_t kMinInlierCount = 15;  // 15个匹配点
constexpr double kActivationThreshold = 1.02; // 1.05 -> 1.02

LinePrimitive FitLinePrimitive(const std::vector<Eigen::Vector2d>& points) {
  LinePrimitive primitive;
  if (points.empty()) {
    return primitive;
  }

  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  for (const auto& pt : points) {
    mean += pt;
  }
  mean /= static_cast<double>(points.size());

  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (const auto& pt : points) {
    Eigen::Vector2d diff = pt - mean;
    cov += diff * diff.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
  Eigen::Vector2d direction = solver.eigenvectors().col(1);
  if (solver.eigenvalues()(1) < 1e-6) {
    direction = Eigen::Vector2d::UnitX();
  }
  direction.normalize();
  double min_proj = std::numeric_limits<double>::max();
  double max_proj = std::numeric_limits<double>::lowest();
  for (const auto& pt : points) {
    double proj = direction.dot(pt - mean);
    min_proj = std::min(min_proj, proj);
    max_proj = std::max(max_proj, proj);
  }
  primitive.center = mean;
  primitive.direction = direction;
  primitive.length = max_proj - min_proj;
  return primitive;
}

DirectedPointPrimitive FitArrowPrimitive(
    const std::vector<Eigen::Vector2d>& points) {
  DirectedPointPrimitive primitive;
  if (points.empty()) {
    return primitive;
  }
  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  for (const auto& pt : points) {
    mean += pt;
  }
  mean /= static_cast<double>(points.size());
  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (const auto& pt : points) {
    Eigen::Vector2d diff = pt - mean;
    cov += diff * diff.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
  Eigen::Vector2d dir = solver.eigenvectors().col(1);
  if (solver.eigenvalues()(1) < 1e-8) {
    dir = Eigen::Vector2d::UnitX();
  }
  dir.normalize();
  primitive.position = mean;
  primitive.direction = dir;
  return primitive;
}
}

// 构造函数：
AvpLocalization::AvpLocalization(std::string map_file) {
  // 初始化前一关键帧时间戳为-1，表示尚未有关键帧
  pre_key_pose_.time_ = -1;

  // 加载地图数据
  avp_map_.load(map_file);
  map_primitives_ = avp_map_.primitives();
  namespace fs = std::filesystem;
  fs::path log_dir;
  try {
    const std::string pkg_path = ros::package::getPath("avp_localization");
    log_dir = fs::path(pkg_path) / "output";
  } catch (const std::exception &e) {
    std::cerr << "Failed to resolve package path: " << e.what() << std::endl;
    log_dir = fs::path("output");
  }
  try {
    fs::create_directories(log_dir);
  } catch (const std::exception &e) {
    std::cerr << "Failed to create log directory: " << log_dir << " "
              << e.what() << std::endl;
  }
  const fs::path plot_root = log_dir / "plots";
  const fs::path quality_plot_dir = plot_root / "03";
  const fs::path primitive_plot_dir = plot_root / "04";
  try {
    fs::create_directories(quality_plot_dir);
  } catch (const std::exception &e) {
    std::cerr << "Failed to create plot directory: " << quality_plot_dir << " "
              << e.what() << std::endl;
  }
  try {
    fs::create_directories(primitive_plot_dir);
  } catch (const std::exception &e) {
    std::cerr << "Failed to create primitive plot directory: "
              << primitive_plot_dir << " " << e.what() << std::endl;
  }
  log_directory_ = log_dir.string();
  semantic_inlier_log_path_ = (log_dir / "semantic_inliers.csv").string();
  residual_log_path_ = (log_dir / "residual_breakdown.csv").string();
  iteration_log_path_ = (log_dir / "iteration_trace.csv").string();
  hessian_log_path_ = (log_dir / "hessian_spectrum.csv").string();
  hessian_iter_log_path_ = (log_dir / "hessian_iter_spectrum.csv").string();
  observation_log_path_ =
      (quality_plot_dir / "observation_quality.csv").string();
  primitive_plot_dir_ = primitive_plot_dir.string();
  map_primitive_export_path_ =
      (primitive_plot_dir / "map_primitives.csv").string();
  primitive_snapshot_path_ =
      (primitive_plot_dir / "primitive_snapshots.csv").string();
  std::cout << "Loaded map primitives: points=" << map_primitives_.totalPointCount()
            << " slot_lines=" << map_primitives_.slot_lines.size()
            << " dash_lines=" << map_primitives_.dash_lines.size()
            << " arrows=" << map_primitives_.arrow_directions.size() << std::endl;

  // 创建output/plots/04目录并初始化语义残差统计CSV文件
  fs::path plot04_dir = log_dir / "plots" / "04";
  try {
    fs::create_directories(plot04_dir);

    const std::string residual_file = (plot04_dir / "semantic_residual_stats.csv").string();
    std::ifstream check_file(residual_file);
    if (!check_file.good()) {
      // 文件不存在，创建并写入header
      std::ofstream ofs(residual_file);
      if (ofs.is_open()) {
        ofs << "time,"
            << "slot_sum_sq,slot_count,slot_rmse,"
            << "dash_sum_sq,dash_count,dash_rmse,"
            << "arrow_sum_sq,arrow_count,arrow_rmse,"
            << "weight_slot,weight_dash,weight_arrow" << std::endl;
        ofs.close();
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Failed to create plot04 directory or CSV file: " << plot04_dir << " "
              << e.what() << std::endl;
  }

  // 创建地图可视化对象
  map_viewer_.reset(new MapViewer(avp_map_));
  adaptive_status_ = AdaptiveStatus{};
  // 初始化车辆坐标系到IPM图像坐标系的变换矩阵
  // 旋转为0
  T_vehicle_ipm_.linear().setIdentity(); 
  // 中心坐标系-->车辆坐标系
  T_vehicle_ipm_.translation() = Eigen::Vector3d(0.0, 1.32, 0);  // y轴为车辆的前进方向
  // T_vehicle_ipm_.translation() = Eigen::Vector3d(0.0, 0.0, 1.32);   // z轴为车辆的前进方向
  // T_vehicle_ipm_.translation() = Eigen::Vector3d(1.32, 0.0, 0.0);   // x轴为车辆的前进方向

  // 创建用于存储车道线、箭头线和停车线点云的智能指针
  // 创建三种语义的点云容器
  // 初始化点云指针，确保使用成员变量而非局部变量
  // 遍历地图中的车道线元素，将点云数据存储到对应的容器中
  for (const auto& element :
       avp_map_.getSemanticElement(SemanticLabel::kDashLine)) {
    pcl::PointXYZ pt;
    pt.x = element.x() * kPixelScale;
    pt.y = element.y() * kPixelScale;
    pt.z = 0;     // 设置z坐标为0（表示所有特征都在地平面上）
    dash_line_cloud_in_->points.push_back(pt);
  }

  // 遍历地图中的箭头元素，将点云数据存储到对应的容器中
  for (const auto& element :
       avp_map_.getSemanticElement(SemanticLabel::kArrowLine)) {
    pcl::PointXYZ pt;
    pt.x = element.x() * kPixelScale;
    pt.y = element.y() * kPixelScale;
    pt.z = 0;  // 设置z坐标为0（表示所有特征都在地平面上）
    arrow_line_cloud_in_->points.push_back(pt);
  }

  // 遍历地图中的停车线元素，将点云数据存储到对应的容器中
  for (const auto& element :
       avp_map_.getSemanticElement(SemanticLabel::kSlot)) {
    pcl::PointXYZ pt;
    pt.x = element.x() * kPixelScale;
    pt.y = element.y() * kPixelScale;
    pt.z = 0;
    slot_cloud_in_->points.push_back(pt);
  }

  /// 初始化KD树
  /// 用提取的特征点建立KD树索引结构
  kdtree_dash_line_.setInputCloud(dash_line_cloud_in_);
  kdtree_arrow_line_.setInputCloud(arrow_line_cloud_in_);
  kdtree_slot_.setInputCloud(slot_cloud_in_);

  if (!map_primitive_export_path_.empty()) {
    try {
      if (!fs::exists(map_primitive_export_path_)) {
        writePrimitiveCsv(map_primitives_, 0.0, "map",
                          map_primitive_export_path_, false);
      }
    } catch (const std::exception &e) {
      std::cerr << "Failed to export map primitives: " << e.what()
                << std::endl;
    }
  }
}


// 获取当前的融合定位结果
TimedPose AvpLocalization::getCurrentPose() const {
  State state = ekf_.getState();
  TimedPose pose;
  pose.time_ = state.time;
  pose.t_ = {state.x, state.y, 0.0};
  pose.R_ = Eigen::Quaterniond(Eigen::AngleAxisd(state.yaw, Eigen::Vector3d::UnitZ()));
  return pose;
}


// 判断一个给定的为位姿是否为关键帧
bool AvpLocalization::isKeyFrame(const TimedPose& pose) {
  // 当前位置与上一关键帧位置的欧几里得距离超过2米：这是一个位移阈值 // 2 -> 1.0 -> 0.8
  if ((pose.t_ - pre_key_pose_.t_).norm() > 1.0 ||
  // 当前朝向与上一关键帧朝向的角度差异超过20度：这是一个角度阈值  // 20 -> 5 -> 3
      std::fabs(GetYaw(pre_key_pose_.R_.inverse() * pose.R_)) > 5. * kToRad ||
  // 前时间与上一关键帧时间间隔超过30秒，或者第一帧：这是一个时间阈值  // 30 -> 0.5 -> 0.3
      pose.time_ - pre_key_pose_.time_ > 0.5 || pre_key_pose_.time_ < -1) {
    // 当前帧（上一帧位姿存储为最新关键帧位姿）    
    pre_key_pose_ = pose;
    return true;  // 返回true表示这是一个关键帧
  }
  return false;
}


// 该函数必须在使用任何定位功能之前调用，为系统提供起点
void AvpLocalization::initState(double time, double x, double y, double yaw) {
  // 需要设置初始的位置！！
  // 创建EKF的实例！
  ekf_ = EKF(time, x, y, yaw);

  // 更新可视化
  if (map_viewer_) {
    // 更新当前位置信息
    curr_frame_.t_update = {x, y, 0};
    // 显示初始位置
    map_viewer_->showFrame(curr_frame_); // init position
  }
}


// 处理来自轮速计的测量数据，将其添加到测量队列中，
// 为后续的状态更新做准备
void AvpLocalization::processWheelMeasurement(
    const WheelMeasurement& wheel_mearsure) {
  wheel_measurements_.push_back(wheel_mearsure);
}


/**
 * 1、传感器融合：整合轮速计数据（运动预测）和视觉数据（观测更新）
 * 2、时间同步：确保所有传感器数据按时间顺序处理
 * 3、关键帧筛选：只处理重要的帧，提高计算效率
 * 4、特征提取：从分割图像中提取地图元素
 * 5、坐标变换：在不同坐标系（世界、车辆、图像）间转换 ？在哪里出现的坐标转换
 * 6、状态更新：使用ICP结果更新EKF状态
 * 7、可视化反馈：提供实时定位结果的视觉反馈
 */
// 处理图像：实现了完整的视觉-轮速融合定位流程，是系统的核心处理环节，
// 通过EKF框架优雅地融合了不同来源的信息，实现了高精度定位。
// not realtime
void AvpLocalization::processImage(double time, const cv::Mat& ipm_seg_img) {
  // do predict -- for sync
  // 处理所有时间戳早于当前图像的轮速数据
  while (!wheel_measurements_.empty() &&
         wheel_measurements_.front().time_ < time) {
    // 暂时回退到原始稳定版本，避免坐标系问题
    ekf_.ekfPredict(wheel_measurements_.front().time_,
                    wheel_measurements_.front().velocity_,
                    wheel_measurements_.front().yaw_rate_);
    // 处理完的数据从队列中移除
    wheel_measurements_.pop_front();
  }
  // 如果队列中还有数据，使用最新的轮速值预测到当前图像的时间点
  // 确保EKF预测步骤在处理图像之前完成
  // 这确保状态预测与图像时间戳精确同步
  if (!wheel_measurements_.empty()) {
    ekf_.ekfPredict(time, wheel_measurements_.front().velocity_,
                    wheel_measurements_.front().yaw_rate_);
  }

  // 只有EKF初始化后才能继续处理
  if (ekf_.isInit()) {
    // 获取当前状态
    State state = ekf_.getState();
    // 创建一个Timepose对象，存储当前时间戳和位置
    TimedPose pose;
    // 设置时间戳
    pose.time_ = state.time;
    // 设置位置
    pose.t_ = {state.x, state.y, 0.0};
    // 设置旋转
    pose.R_ = Eigen::Quaterniond(
        Eigen::AngleAxisd{state.yaw, Eigen::Vector3d::UnitZ()});

    // 判断当前位姿是否为关键帧
    if (!isKeyFrame(pose)) {
      adaptive_status_.stamp = time;
      adaptive_status_.measurement_valid = false;
      adaptive_status_.phi = ekf_.last_phi();
      adaptive_status_.gamma = ekf_.last_gamma();
      adaptive_status_.quality = ekf_.last_quality_score();
      adaptive_status_.mode = AdaptiveMode::kIdle;
      return;
    }
    // 打印关键帧信息
    std::cout << std::endl;
    printf("Keyframe: time = %.4f  (%.3f, %.3f, %.2f) \n", time, state.x,
           state.y, state.yaw * kToDeg);
    // 将IPM图像转换为灰度图像
    cv::Mat img_gray;
    cv::cvtColor(ipm_seg_img, img_gray, cv::COLOR_BGR2GRAY);

    // 测试点1
    printf("test point 1 !\n");

    // 提取语义特征：车道线、箭头、停车线（使用当前的位姿）
    extractFeature(img_gray, pose);
    const PrimitiveSet pre_primitives_snapshot = curr_frame_.primitives();

    // 测试点2
    printf("test point 2 !\n");

    // 图像配准：将当前帧的特征点云和地图中的特征点云进行配准（ICP）
    RegistrationResult registration = imageRegistration(time);
    Eigen::Affine3d T_world_vehicle = registration.T_world_vehicle;
    // 获取配准后的位置向量3x1
    Eigen::Vector3d t_est = T_world_vehicle.translation();
    // 获取配准后的旋转矩阵3x3
    Eigen::Matrix3d R_est = T_world_vehicle.rotation();
    // 输出ICP估计的位置和角度
    printf("ICP: time = %.4f  (%.3f, %.3f, %.2f) \n", time, t_est.x(),
           t_est.y(), GetYaw(Eigen::Quaterniond(R_est)) * kToDeg);
    
    // 测试点3
    printf("test point 3 !\n");

    // 更新EKF状态（视觉定位作为观测）
    bool measurement_valid =
        registration.inlier_count >= kMinInlierCount &&
        std::isfinite(registration.fitness);
    ekf_.ekfUpdate(t_est.x(), t_est.y(), GetYaw(Eigen::Quaterniond(R_est)),
                 registration.fitness, registration.inlier_ratio,
                 measurement_valid);
    adaptive_status_.stamp = time;
    adaptive_status_.measurement_valid = measurement_valid;
    adaptive_status_.phi = ekf_.last_phi();
    adaptive_status_.gamma = ekf_.last_gamma();
    adaptive_status_.quality = ekf_.last_quality_score();
    adaptive_status_.fitness = registration.fitness;
    adaptive_status_.inlier_ratio = registration.inlier_ratio;
    const bool strong_tracking =
        measurement_valid && adaptive_status_.phi > kActivationThreshold;
    const bool observation_downweight =
        measurement_valid && adaptive_status_.gamma > kActivationThreshold;
    if (!measurement_valid) {
      adaptive_status_.mode = AdaptiveMode::kIdle;
    } else if (strong_tracking && observation_downweight) {
      adaptive_status_.mode = AdaptiveMode::kCombined;
    } else if (strong_tracking) {
      adaptive_status_.mode = AdaptiveMode::kStrongTracking;
    } else if (observation_downweight) {
      adaptive_status_.mode = AdaptiveMode::kObservationDownweight;
    } else {
      adaptive_status_.mode = AdaptiveMode::kNormal;
    }
    logObservationQuality(time, registration, adaptive_status_);
    // 可视化更新
    std::cout << "**********************************************" << std::endl;
    state = ekf_.getState(); // 获取更新后的EKF状态
    TimedPose updated_pose;
    updated_pose.time_ = state.time;
    updated_pose.t_ = {state.x, state.y, 0.0};
    updated_pose.R_ = Eigen::Quaterniond(
        Eigen::AngleAxisd{state.yaw, Eigen::Vector3d::UnitZ()});

    // 再次提取特征（使用更新后的位姿）
    extractFeature(img_gray, updated_pose);
    exportFramePrimitiveSets(time, pre_primitives_snapshot,
                             curr_frame_.primitives());
    // 更新当前帧的位姿
    curr_frame_.t_update = updated_pose.t_;
    if (map_viewer_) {
      // 调用地图可视化显示当前帧
      map_viewer_->showFrame(curr_frame_);
    }

  } else {
    std::cout << "Unable to get state at " << std::to_string(time) << std::endl;
    adaptive_status_.stamp = time;
    adaptive_status_.measurement_valid = false;
    adaptive_status_.mode = AdaptiveMode::kIdle;
  }
}


// 将IPM图像中像素坐标转换为世界坐标系下的3D坐标
Eigen::Vector3d AvpLocalization::ipmPlane2Global(
    const TimedPose& T_world_vehicle, const cv::Point2f& ipm_point) {
  Eigen::Vector3d pt_global;
  Eigen::Affine3d T_world_ipm = Eigen::Translation3d(T_world_vehicle.t_) *
                                T_world_vehicle.R_ * T_vehicle_ipm_;
  Eigen::Vector3d pt_ipm{-(0.5 * kIPMImgHeight - ipm_point.x) * kPixelScale,
                         (0.5 * kIPMImgWidth - ipm_point.y) * kPixelScale, 0.0};
  pt_global = T_world_ipm * pt_ipm;
  return pt_global;
}


// 从IPM图像中提取三种语义特征，并将这些特征转换到世界坐标系
// 1、车道线
// 2、箭头线
// 3、停车线
void AvpLocalization::extractFeature(const cv::Mat& img_gray,
                                     const TimedPose& T_world_vehicle) {
  // 清除当前帧的语义元素网格地图，准备存储新特征
  curr_frame_.clearGridMap();
  PrimitiveSet& primitives = curr_frame_.primitives();
  // 计算从世界坐标系到IPM图像坐标系的变换矩阵？？
  curr_frame_.T_world_ipm_ = Eigen::Translation3d(T_world_vehicle.t_) *
                             T_world_vehicle.R_ * T_vehicle_ipm_;
  // 专门用于停车位线特征提取
  cv::Mat slot_img = cv::Mat::zeros(img_gray.size(), img_gray.type());
  // 用于箭头线和车道虚线特征提取
  cv::Mat seg_img = cv::Mat::zeros(img_gray.size(), img_gray.type());
  // 使用预定义的灰度常量（kSlotGray等）识别不同语义类别
  // 将对应像素设置为254（接近白色），形成二值化特征掩膜
  for (int i = 0; i < img_gray.rows; ++i) {
    for (int j = 0; j < img_gray.cols; ++j) {
      if (kSlotGray == img_gray.at<uchar>(i, j) ||
          kSlotGray1 == img_gray.at<uchar>(i, j)) {
        // 将slot_img中对应位置像素设置为254（白色）
        slot_img.at<uchar>(i, j) = 254;
      } else if (kArrowGray == img_gray.at<uchar>(i, j)) {
        // 将seg_img中对应位置像素设置为254（白色）
        seg_img.at<uchar>(i, j) = 254;
      } else if (kDashGray == img_gray.at<uchar>(i, j)) {
        seg_img.at<uchar>(i, j) = 254;
      }
    }
  }

  /// 处理停车位线特征
  // 形态学闭操作（去噪）
  int kernelSize = 13;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                             cv::Size(kernelSize, kernelSize));
  cv::Mat closed;
  cv::morphologyEx(slot_img, closed, cv::MORPH_CLOSE, kernel);
  auto line_img = closed.clone();

  // 骨架提取
  cv::Mat skel = skeletonize(closed);
  removeIsolatedPixels(skel, 2);

  std::vector<std::vector<cv::Point>> slot_contours;
  std::vector<cv::Vec4i> slot_hierarchy;
  cv::findContours(skel, slot_contours, slot_hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);
  for (const auto& contour : slot_contours) {
    if (contour.size() < 6) {
      continue;
    }
    std::vector<Eigen::Vector2d> global_points;
    global_points.reserve(contour.size());
    for (const auto& pt : contour) {
      Eigen::Vector3d global =
          ipmPlane2Global(T_world_vehicle, cv::Point2f(pt.x, pt.y));
      curr_frame_.addSemanticElement(SemanticLabel::kSlot, global);
      primitives.slot_points.emplace_back(global.x(), global.y());
      global_points.emplace_back(global.x(), global.y());
    }
    LinePrimitive primitive = FitLinePrimitive(global_points);
    if (primitive.length > 0.2) {
      primitives.slot_lines.push_back(primitive);
    }
  }

  // 处理箭头/虚线特征
  // find contour
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  // 检测外部轮廓，使用简单近似（减少点数） --> 提取出轮廓点
  cv::findContours(seg_img, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  
  // 将轮廓点坐标转换到全局坐标系
  for (const auto& contour : contours) {
    std::vector<Eigen::Vector2d> dash_points;
    std::vector<Eigen::Vector2d> arrow_points;
    for (const auto& point : contour) {
      const auto pixel_value = img_gray.at<uchar>(point);
      if (pixel_value != kArrowGray && pixel_value != kDashGray) {
        continue;
      }
      Eigen::Vector3d global =
          ipmPlane2Global(T_world_vehicle, cv::Point2f(point.x, point.y));
      if (pixel_value == kArrowGray) {
        curr_frame_.addSemanticElement(SemanticLabel::kArrowLine, global);
        arrow_points.emplace_back(global.x(), global.y());
        primitives.arrow_points.emplace_back(global.x(), global.y());
      } else {
        curr_frame_.addSemanticElement(SemanticLabel::kDashLine, global);
        dash_points.emplace_back(global.x(), global.y());
        primitives.dash_points.emplace_back(global.x(), global.y());
      }
    }
    if (dash_points.size() >= 4) {
      LinePrimitive primitive = FitLinePrimitive(dash_points);
      if (primitive.length > 0.15) {
        primitives.dash_lines.push_back(primitive);
      }
    }
    if (arrow_points.size() >= 3) {
      primitives.arrow_directions.push_back(FitArrowPrimitive(arrow_points));
    }
  }

  std::cout << "Frame primitives: slot_pts=" << primitives.slot_points.size()
            << " dash_pts=" << primitives.dash_points.size()
            << " arrow_pts=" << primitives.arrow_points.size()
            << " slot_lines=" << primitives.slot_lines.size()
            << " dash_lines=" << primitives.dash_lines.size()
            << " arrows=" << primitives.arrow_directions.size() << std::endl;
}

// 图像配准：将当前帧的特征点云与地图中的特征点云进行匹配
// Registration frame to map

RegistrationResult AvpLocalization::imageRegistration(double time) {
  RegistrationResult result;
  const PrimitiveSet& frame_primitives = curr_frame_.primitives();

  SemanticCounts semantic_counts;
  semantic_counts.slot_sampled =
      frame_primitives.slot_points.size() + frame_primitives.slot_lines.size();
  semantic_counts.dash_sampled =
      frame_primitives.dash_points.size() + frame_primitives.dash_lines.size();
  semantic_counts.arrow_sampled =
      frame_primitives.arrow_points.size() +
      frame_primitives.arrow_directions.size();

  std::size_t total_features = frame_primitives.totalPointCount();
  total_features += frame_primitives.slot_lines.size();
  total_features += frame_primitives.dash_lines.size();
  total_features += frame_primitives.arrow_directions.size();
  result.total_count = total_features;
  if (total_features == 0) {
    result.T_world_vehicle = Eigen::Affine3d::Identity();
    return result;
  }

  const PrimitiveSet& map_primitives = map_primitives_;
  if (map_primitives.totalPointCount() == 0 &&
      map_primitives.slot_lines.empty() &&
      map_primitives.dash_lines.empty()) {
    result.T_world_vehicle = Eigen::Affine3d::Identity();
    return result;
  }

  auto findNearestPointWithTree =
      [&](const Eigen::Vector2d& query,
          const pcl::KdTreeFLANN<pcl::PointXYZ>& tree,
          const pcl::PointCloud<pcl::PointXYZ>::Ptr& map_cloud,
          Eigen::Vector2d* nearest, double max_dist_sq) -> bool {
    pcl::PointXYZ search_point;
    search_point.x = static_cast<float>(query.x());
    search_point.y = static_cast<float>(query.y());
    search_point.z = 0.0f;
    std::vector<int> indices(1);
    std::vector<float> distances(1);
    const int found = tree.nearestKSearch(search_point, 1, indices, distances);
    if (found > 0 && distances[0] < max_dist_sq &&
        indices[0] < static_cast<int>(map_cloud->size())) {
      const auto& pt = map_cloud->points[indices[0]];
      *nearest = Eigen::Vector2d(pt.x, pt.y);
      return true;
    }
    return false;
  };

  auto findClosestLine = [](const std::vector<LinePrimitive>& candidates,
                            const LinePrimitive& query)
                              -> const LinePrimitive* {
    const LinePrimitive* best = nullptr;
    double best_score = std::numeric_limits<double>::max();
    for (const auto& line : candidates) {
      const double center_dist = (line.center - query.center).norm();
      const double orientation_error =
          1.0 - std::fabs(line.direction.dot(query.direction));
      const double score = center_dist + orientation_error;
      if (score < best_score && center_dist < 1.5) {
        best = &line;
        best_score = score;
      }
    }
    return best;
  };

  Eigen::Vector2d translation = Eigen::Vector2d::Zero();
  double rotation = 0.0;
  Eigen::Matrix2d skew;
  skew << 0.0, -1.0, 1.0, 0.0;

  ResidualStats slot_stats;
  ResidualStats dash_stats;
  ResidualStats arrow_stats;

  std::vector<IterationLog> iteration_logs;
  iteration_logs.reserve(10);

  std::vector<HessianIterLog> hessian_logs;
  hessian_logs.reserve(10);

  int final_residual_count = 0;
  Eigen::Matrix3d last_H = Eigen::Matrix3d::Zero();

  const double kSlotDistanceSq = 0.6 * 0.6;
  const double kDashDistanceSq = 0.5 * 0.5;
  const double kArrowDistanceSq = 0.7 * 0.7;

  const int kMaxIter = 10;
  for (int iter = 0; iter < kMaxIter; ++iter) {
    const Eigen::Rotation2Dd R(rotation);
    Eigen::Matrix2d R_mat = R.toRotationMatrix();

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    double residual_sum = 0.0;
    int residual_count = 0;
    std::size_t used_terms = 0;

    auto accumulatePointResidual = [&](const Eigen::Vector2d& point,
                                       const Eigen::Vector2d& map_pt,
                                       double weight, ResidualStats* stats) {
      const Eigen::Vector2d transformed = R_mat * point + translation;
      const Eigen::Vector2d residual = transformed - map_pt;
      if (stats) {
        stats->sum_sq += residual.squaredNorm();
        stats->count += 1;
      }
      Eigen::Matrix<double, 2, 3> J;
      J.setZero();
      J.block<2, 2>(0, 0).setIdentity();
      J.col(2) = R_mat * (skew * point);
      H += weight * J.transpose() * J;
      g += weight * J.transpose() * residual;
      residual_sum += weight * residual.squaredNorm();
      residual_count += 1;
      ++used_terms;
    };

    auto accumulateLineResidual = [&](const LinePrimitive& frame_line,
                                      const LinePrimitive& map_line,
                                      double weight, ResidualStats* stats) {
      Eigen::Vector2d normal(-map_line.direction.y(), map_line.direction.x());
      normal.normalize();
      const Eigen::Vector2d transformed =
          R_mat * frame_line.center + translation;
      const double residual = normal.dot(transformed - map_line.center);
      if (stats) {
        stats->sum_sq += residual * residual;
        stats->count += 1;
      }
      Eigen::RowVector3d J = Eigen::RowVector3d::Zero();
      J.head<2>() = normal.transpose();
      J(2) = normal.dot(R_mat * (skew * frame_line.center));
      H += weight * J.transpose() * J;
      g += weight * J.transpose() * residual;
      residual_sum += weight * residual * residual;
      ++residual_count;
      ++used_terms;

      const Eigen::Vector2d rotated_dir = R_mat * frame_line.direction;
      const double sin_diff =
          map_line.direction.x() * rotated_dir.y() -
          map_line.direction.y() * rotated_dir.x();
      const double orient_weight = weight * 0.25;
      Eigen::RowVector3d J_ori = Eigen::RowVector3d::Zero();
      J_ori(2) = map_line.direction.dot(R_mat * (skew * frame_line.direction));
      H += orient_weight * J_ori.transpose() * J_ori;
      g += orient_weight * J_ori.transpose() * sin_diff;
      residual_sum += orient_weight * sin_diff * sin_diff;
      ++residual_count;
    };

    for (const auto& point : frame_primitives.slot_points) {
      Eigen::Vector2d map_pt;
      if (!findNearestPointWithTree(point, kdtree_slot_, slot_cloud_in_, &map_pt,
                                    kSlotDistanceSq)) {
        continue;
      }
      semantic_counts.slot_used++;
      accumulatePointResidual(point, map_pt, 2.0, &slot_stats);
    }

    for (const auto& point : frame_primitives.dash_points) {
      Eigen::Vector2d map_pt;
      if (!findNearestPointWithTree(point, kdtree_dash_line_, dash_line_cloud_in_,
                                    &map_pt, kDashDistanceSq)) {
        continue;
      }
      semantic_counts.dash_used++;
      accumulatePointResidual(point, map_pt, 1.0, &dash_stats);
    }

    for (const auto& point : frame_primitives.arrow_points) {
      Eigen::Vector2d map_pt;
      if (!findNearestPointWithTree(point, kdtree_arrow_line_, arrow_line_cloud_in_,
                                    &map_pt, kArrowDistanceSq)) {
        continue;
      }
      semantic_counts.arrow_used++;
      accumulatePointResidual(point, map_pt, 1.2, &arrow_stats);
    }

    for (const auto& line : frame_primitives.slot_lines) {
      const LinePrimitive* map_line =
          findClosestLine(map_primitives.slot_lines, line);
      if (!map_line) {
        continue;
      }
      semantic_counts.slot_used++;
      const double weight = 2.0 * std::max(line.length, 0.1);
      accumulateLineResidual(line, *map_line, weight, &slot_stats);
    }

    for (const auto& line : frame_primitives.dash_lines) {
      const LinePrimitive* map_line =
          findClosestLine(map_primitives.dash_lines, line);
      if (!map_line) {
        continue;
      }
      semantic_counts.dash_used++;
      const double weight = 1.0 * std::max(line.length, 0.1);
      accumulateLineResidual(line, *map_line, weight, &dash_stats);
    }

    if (!frame_primitives.arrow_directions.empty() &&
        !map_primitives.arrow_directions.empty()) {
      const auto& frame_arrow = frame_primitives.arrow_directions.front();
      const auto& map_arrow = map_primitives.arrow_directions.front();
      Eigen::Vector2d normal(-map_arrow.direction.y(), map_arrow.direction.x());
      normal.normalize();
      const Eigen::Vector2d transformed =
          R_mat * frame_arrow.position + translation;
      const double residual = normal.dot(transformed - map_arrow.position);
      semantic_counts.arrow_used++;
      arrow_stats.sum_sq += residual * residual;
      arrow_stats.count += 1;
      Eigen::RowVector3d J = Eigen::RowVector3d::Zero();
      J.head<2>() = normal.transpose();
      J(2) = normal.dot(R_mat * (skew * frame_arrow.position));
      const double weight = 1.0;
      H += weight * J.transpose() * J;
      g += weight * J.transpose() * residual;
      residual_sum += weight * residual * residual;
      ++residual_count;
      ++used_terms;

      const Eigen::Vector2d rotated_dir = R_mat * frame_arrow.direction;
      const double sin_diff =
          map_arrow.direction.x() * rotated_dir.y() -
          map_arrow.direction.y() * rotated_dir.x();
      Eigen::RowVector3d J_ori = Eigen::RowVector3d::Zero();
      J_ori(2) =
          map_arrow.direction.dot(R_mat * (skew * frame_arrow.direction));
      const double orient_weight = 0.5;
      H += orient_weight * J_ori.transpose() * J_ori;
      g += orient_weight * J_ori.transpose() * sin_diff;
      residual_sum += orient_weight * sin_diff * sin_diff;
      ++residual_count;
    }

    hessian_logs.push_back({iter, H, residual_count});
    
    // ICP终止条件1：若当前迭代没有任何匹配项贡献 Hessian
    // 要么找不到对应的语义原语、要么残差被全部过滤，算法立即退出
    if (used_terms == 0 || residual_count == 0) {
      last_H = H;
      final_residual_count = residual_count;
      iteration_logs.push_back({iter, 0.0, residual_sum});
      break;
    }

    Eigen::Vector3d delta = Eigen::Vector3d::Zero();
    Eigen::LDLT<Eigen::Matrix3d> solver(H);
    if (solver.isPositive()) {
      delta = solver.solve(-g);
    } else {
      delta = H.ldlt().solve(-g);
    }

    // ICP终止条件2：线性求解得到的三维增量包含 NaN/Inf
    // 例如 Hessian 病态或数值溢出，马上停止，防止把无效姿态写回
    if (!delta.allFinite()) {
      last_H = H;
      final_residual_count = residual_count;
      hessian_logs.back().H = H;
      hessian_logs.back().residual_count = residual_count;
      iteration_logs.push_back({iter, 0.0, residual_sum});
      break;
    }

    const double delta_norm = delta.norm();
    iteration_logs.push_back({iter, delta_norm, residual_sum});

    translation += delta.head<2>();
    rotation += delta(2);
    last_H = H;
    final_residual_count = residual_count;
    hessian_logs.back().H = H;
    hessian_logs.back().residual_count = residual_count;

    // ICP终止条件3：当迭代增量的欧氏范数小于 1e‑4 即认为收敛
    // 参数调整：1e-4 -> 1e-5 -> 1e-6 -> 1e-7
    if (delta_norm < 1e-4) {
      break;
    }
  }

  semantic_counts.slot_final = semantic_counts.slot_used;
  semantic_counts.dash_final = semantic_counts.dash_used;
  semantic_counts.arrow_final = semantic_counts.arrow_used;

  const std::size_t final_inlier_count = semantic_counts.slot_final +
                                         semantic_counts.dash_final +
                                         semantic_counts.arrow_final;

  Eigen::Affine3d dT =
      Eigen::Translation3d(translation.x(), translation.y(), 0.0) *
      Eigen::AngleAxisd(rotation, Eigen::Vector3d::UnitZ());
  Eigen::Affine3d T_world_vehicle =
      dT * curr_frame_.T_world_ipm_ * T_vehicle_ipm_.inverse();

  const double total_sum_sq = slot_stats.sum_sq + dash_stats.sum_sq +
                              arrow_stats.sum_sq;
  const std::size_t total_sum_count =
      slot_stats.count + dash_stats.count + arrow_stats.count;

  double fitness = std::numeric_limits<double>::infinity();
  if (total_sum_count > 0) {
    fitness = std::sqrt(total_sum_sq /
                        static_cast<double>(total_sum_count));
  }

  result.inlier_count = final_inlier_count;
  result.inlier_ratio =
      (total_features > 0)
          ? static_cast<double>(final_inlier_count) /
                static_cast<double>(total_features)
          : 0.0;
  result.fitness = fitness;
  result.T_world_vehicle = T_world_vehicle;

  if (total_features > 0) {
    logSemanticInliers(time, semantic_counts, total_features);
    logResidualBreakdown(time, slot_stats, dash_stats, arrow_stats);
    if (!primitive_plot_dir_.empty()) {
      namespace fs = std::filesystem;
      const fs::path residual_path =
          fs::path(primitive_plot_dir_) / "semantic_residual_stats.csv";
      std::ofstream ofs(residual_path, std::ios::app);
      if (ofs.is_open()) {
        ofs << std::fixed << std::setprecision(6)
            << time << ',' << slot_stats.sum_sq << ',' << slot_stats.count
            << ','
            << (slot_stats.count > 0
                    ? std::sqrt(slot_stats.sum_sq / slot_stats.count)
                    : 0.0)
            << ',' << dash_stats.sum_sq << ',' << dash_stats.count << ','
            << (dash_stats.count > 0
                    ? std::sqrt(dash_stats.sum_sq / dash_stats.count)
                    : 0.0)
            << ',' << arrow_stats.sum_sq << ',' << arrow_stats.count << ','
            << (arrow_stats.count > 0
                    ? std::sqrt(arrow_stats.sum_sq / arrow_stats.count)
                    : 0.0)
            << ",2.0,1.0,1.2\n";
      } else {
        std::cerr << "Failed to open semantic residual stats log: "
                  << residual_path << std::endl;
      }
    }

    logIterationTrace(time, iteration_logs);
    logHessianIterSpectrum(time, hessian_logs);
    if (final_residual_count > 0) {
      logHessianSpectrum(time, last_H, final_residual_count);
    }
  }

  return result;
}

void AvpLocalization::appendCsv(const std::string &path,
                                const std::string &header,
                                const std::string &row) const {
  if (path.empty()) {
    return;
  }
  namespace fs = std::filesystem;
  const fs::path file_path(path);
  const bool write_header = !fs::exists(file_path);
  std::ofstream ofs(path, std::ios::app);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open log file: " << path << std::endl;
    return;
  }
  if (write_header) {
    ofs << header << '\n';
  }
  ofs << row << '\n';
}

void AvpLocalization::logSemanticInliers(double time,
                                         const SemanticCounts &counts,
                                         std::size_t total_features) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << time << ',' << counts.slot_sampled
      << ',' << counts.slot_used << ',' << counts.slot_final << ','
      << counts.dash_sampled << ',' << counts.dash_used << ','
      << counts.dash_final << ',' << counts.arrow_sampled << ','
      << counts.arrow_used << ',' << counts.arrow_final << ','
      << total_features;
  appendCsv(semantic_inlier_log_path_,
            "time,slot_sampled,slot_used,slot_final,"
            "dash_sampled,dash_used,dash_final,"
            "arrow_sampled,arrow_used,arrow_final,total_features",
            oss.str());
}

void AvpLocalization::logResidualBreakdown(double time,
                                           const ResidualStats &slot_stats,
                                           const ResidualStats &dash_stats,
                                           const ResidualStats &arrow_stats) {
  auto to_rmse = [](const ResidualStats &stats) {
    if (stats.count == 0) {
      return 0.0;
    }
    return std::sqrt(std::max(stats.sum_sq, 0.0) /
                     static_cast<double>(stats.count));
  };

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << time << ',' << to_rmse(slot_stats)
      << ',' << to_rmse(dash_stats) << ',' << to_rmse(arrow_stats) << ','
      << slot_stats.count << ',' << dash_stats.count << ','
      << arrow_stats.count;
  appendCsv(residual_log_path_,
            "time,rmse_slot_pos,rmse_dash_pos,rmse_arrow_pos,"
            "count_slot,count_dash,count_arrow",
            oss.str());
}

void AvpLocalization::logIterationTrace(double time,
                                        const std::vector<IterationLog> &logs) {
  if (logs.empty()) {
    return;
  }
  for (const auto &entry : logs) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << time << ',' << entry.iter
        << ',' << entry.delta_norm << ',' << entry.residual_sum;
    appendCsv(iteration_log_path_,
              "time,iter,delta_norm,residual_sum",
              oss.str());
  }
}

void AvpLocalization::logHessianSpectrum(double time,
                                         const Eigen::Matrix3d &H,
                                         int residual_count) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(H);
  if (solver.info() != Eigen::Success) {
    return;
  }
  const auto eigenvalues = solver.eigenvalues();
  const double lambda_min = eigenvalues(0);
  const double lambda_mid = eigenvalues(1);
  const double lambda_max = eigenvalues(2);
  const double min_abs = std::max(std::abs(lambda_min), 1e-9);
  const double cond = std::abs(lambda_max) / min_abs;

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << time << ',' << lambda_min << ','
      << lambda_mid << ',' << lambda_max << ',' << cond << ','
      << residual_count;
  appendCsv(hessian_log_path_,
            "time,lambda_min,lambda_mid,lambda_max,condition,residual_count",
            oss.str());
}

void AvpLocalization::logHessianIterSpectrum(
    double time, const std::vector<HessianIterLog> &logs) {
  if (logs.empty()) {
    return;
  }
  for (const auto &entry : logs) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(entry.H);
    if (solver.info() != Eigen::Success) {
      continue;
    }
    const auto eigenvalues = solver.eigenvalues();
    const double lambda_min = eigenvalues(0);
    const double lambda_mid = eigenvalues(1);
    const double lambda_max = eigenvalues(2);
    const double min_abs = std::max(std::abs(lambda_min), 1e-9);
    const double cond = std::abs(lambda_max) / min_abs;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << time << ',' << entry.iter << ','
        << lambda_min << ',' << lambda_mid << ',' << lambda_max << ',' << cond
        << ',' << entry.residual_count;
    appendCsv(hessian_iter_log_path_,
              "time,iter,lambda_min,lambda_mid,lambda_max,condition,residual_count",
              oss.str());
  }
}

void AvpLocalization::logObservationQuality(
    double time, const RegistrationResult &result,
    const AdaptiveStatus &status) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << time << ',' << result.fitness
      << ',' << result.inlier_ratio << ',' << result.inlier_count << ','
      << status.phi << ',' << status.gamma << ',' << status.quality << ','
      << (status.measurement_valid ? 1 : 0) << ','
      << static_cast<int>(status.mode);
  appendCsv(observation_log_path_,
            "time,fitness,inlier_ratio,inlier_count,phi,gamma,quality,"
            "measurement_valid,mode",
            oss.str());
}

void AvpLocalization::exportFramePrimitiveSets(double time,
                                               const PrimitiveSet &pre,
                                               const PrimitiveSet &post) {
  if (primitive_snapshot_path_.empty()) {
    return;
  }
  namespace fs = std::filesystem;
  writePrimitiveCsv(pre, time, "pre", primitive_snapshot_path_, true);
  writePrimitiveCsv(post, time, "post", primitive_snapshot_path_, true);

  if (!map_primitive_export_path_.empty() &&
      !fs::exists(map_primitive_export_path_)) {
    writePrimitiveCsv(map_primitives_, 0.0, "map", map_primitive_export_path_,
                      false);
  }
}

void AvpLocalization::writePrimitiveCsv(const PrimitiveSet &primitives,
                                        double time,
                                        const std::string &set_label,
                                        const std::string &path,
                                        bool append) const {
  if (path.empty()) {
    return;
  }
  namespace fs = std::filesystem;
  const fs::path file_path(path);
  if (file_path.has_parent_path()) {
    try {
      fs::create_directories(file_path.parent_path());
    } catch (const std::exception &e) {
      std::cerr << "Failed to create primitive directory: " << e.what()
                << std::endl;
      return;
    }
  }
  const bool file_exists = fs::exists(file_path);
  std::ofstream ofs(path, append ? std::ios::app : std::ios::out);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open primitive csv: " << path << std::endl;
    return;
  }
  ofs << std::fixed << std::setprecision(6);
  if (!append || !file_exists) {
    ofs << "time,set,label,geom,x,y,length,dir_x,dir_y\n";
  }
  auto emit_point = [&](const std::vector<Eigen::Vector2d> &pts,
                        const std::string &label) {
    for (const auto &pt : pts) {
      ofs << time << ',' << set_label << ',' << label << ",point," << pt.x()
          << ',' << pt.y() << ",,,\n";
    }
  };
  auto emit_line = [&](const std::vector<LinePrimitive> &lines,
                       const std::string &label) {
    for (const auto &line : lines) {
      ofs << time << ',' << set_label << ',' << label << ",line,"
          << line.center.x() << ',' << line.center.y() << ',' << line.length
          << ',' << line.direction.x() << ',' << line.direction.y() << '\n';
    }
  };
  auto emit_arrow_dir =
      [&](const std::vector<DirectedPointPrimitive> &arrows,
          const std::string &label) {
        for (const auto &arrow : arrows) {
          ofs << time << ',' << set_label << ',' << label << ",arrow_dir,"
              << arrow.position.x() << ',' << arrow.position.y() << ",,"
              << arrow.direction.x() << ',' << arrow.direction.y() << '\n';
        }
      };

  emit_point(primitives.slot_points, "slot");
  emit_point(primitives.dash_points, "dash");
  emit_point(primitives.arrow_points, "arrow");
  emit_line(primitives.slot_lines, "slot");
  emit_line(primitives.dash_lines, "dash");
  emit_arrow_dir(primitives.arrow_directions, "arrow");
}
