#pragma once
#include <pcl/kdtree/kdtree_flann.h>  // 点云的kdtree
#include <pcl/point_cloud.h>          // 点云
#include <pcl/point_types.h>          // 点云类型

#include <Eigen/Geometry>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ekf.h"          // 扩展卡尔曼滤波器
#include "frame.h"        // 帧
#include "map.h"          // 地图
#include "map_viewer.h"   // 地图可视化

#include <nav_msgs/OccupancyGrid.h> // 新增头文件，用于发布OccupancyGrid消息

// 时间戳和位姿结构体
struct TimedPose {  
  double time_;
  Eigen::Vector3d t_;
  Eigen::Quaterniond R_;
};

// 轮速测量结构体
struct WheelMeasurement {
  double time_;
  double velocity_;
  double yaw_rate_;
  double steering_angle_{0.0};  // 转向角 (新增，默认为0)
};

struct RegistrationResult {
  Eigen::Affine3d T_world_vehicle;
  double fitness{std::numeric_limits<double>::infinity()};
  double inlier_ratio{0.0};
  std::size_t inlier_count{0};
  std::size_t total_count{0};
};

enum class AdaptiveMode : uint8_t {
  kIdle = 0,
  kNormal = 1,
  kStrongTracking = 2,
  kObservationDownweight = 3,
  kCombined = 4
};

struct AdaptiveStatus {
  double stamp{0.0};
  double phi{1.0};
  double gamma{1.0};
  double quality{0.0};
  double fitness{std::numeric_limits<double>::quiet_NaN()};
  double inlier_ratio{0.0};
  bool measurement_valid{false};
  AdaptiveMode mode{AdaptiveMode::kIdle};
};

// 车辆定位类
class AvpLocalization {
 

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 添加这一行

  // 构造函数
  AvpLocalization(std::string map_file);

  // 析构函数
  ~AvpLocalization() = default;

  // 处理轮速测量
  void processWheelMeasurement(const WheelMeasurement &wheel_measurement);

  // 处理图像
  void processImage(double time, const cv::Mat &ipm_seg_img);

  // 初始化状态
  void initState(double time, double x, double y, double yaw);

  // 图像
  RegistrationResult imageRegistration(double time);

  // 获取地图
  const Map &getMap() const { return avp_map_; }
  //  void setViewer(std::shared_ptr<ViewerInterface> viewer) { viewer_ =
  //  viewer; }

  TimedPose getCurrentPose() const; // 新增方法，用于获取当前位姿
  AdaptiveStatus getAdaptiveStatus() const { return adaptive_status_; }

 private:
  // 判断是否为关键帧
  bool isKeyFrame(const TimedPose &pose);

  // 将图像点转换为全局坐标
  Eigen::Vector3d ipmPlane2Global(const TimedPose &T_world_vehicle,
                                  const cv::Point2f &ipm_point);

  // 提取特征
  void extractFeature(const cv::Mat &img_gray,
                      const TimedPose &T_world_vehicle);

  struct ResidualStats {
    double sum_sq{0.0};
    std::size_t count{0};
  };

  struct SemanticCounts {
    std::size_t slot_sampled{0};
    std::size_t slot_used{0};
    std::size_t slot_final{0};
    std::size_t dash_sampled{0};
    std::size_t dash_used{0};
    std::size_t dash_final{0};
    std::size_t arrow_sampled{0};
    std::size_t arrow_used{0};
    std::size_t arrow_final{0};
  };

  struct IterationLog {
    int iter{0};
    double delta_norm{0.0};
    double residual_sum{0.0};
  };

  struct HessianIterLog {
    int iter{0};
    Eigen::Matrix3d H{Eigen::Matrix3d::Zero()};
    int residual_count{0};
  };

  void logSemanticInliers(double time, const SemanticCounts &counts,
                          std::size_t total_features);
  void logResidualBreakdown(double time, const ResidualStats &slot_stats,
                            const ResidualStats &dash_stats,
                            const ResidualStats &arrow_stats);
  void logIterationTrace(double time, const std::vector<IterationLog> &logs);
  void logHessianSpectrum(double time, const Eigen::Matrix3d &H,
                          int residual_count);
  void logHessianIterSpectrum(double time,
                              const std::vector<HessianIterLog> &logs);
  void logObservationQuality(double time, const RegistrationResult &result,
                             const AdaptiveStatus &status);
  void exportFramePrimitiveSets(double time, const PrimitiveSet &pre,
                                const PrimitiveSet &post);
  void writePrimitiveCsv(const PrimitiveSet &primitives, double time,
                         const std::string &set_label, const std::string &path,
                         bool append) const;
  void appendCsv(const std::string &path, const std::string &header,
                 const std::string &row) const;

  // 临时先用pcl的kdtree，后续有时间可以换成直接依赖FLANN
  // 用于车道线的KD树:快速查找最近的车道线点（支持最近邻搜索）
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_dash_line_;
  // 用于箭头线的KD树：快速查找最近的箭头线点
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_arrow_line_;
  // 用于停车线的KD树：快速查找最近的停车线点
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_slot_;

  // 车道线的点云：用于存储车道线的点云数据
  pcl::PointCloud<pcl::PointXYZ>::Ptr dash_line_cloud_in_{
      new pcl::PointCloud<pcl::PointXYZ>};
  // 箭头线的点云：用于存储箭头线的点云数据
  pcl::PointCloud<pcl::PointXYZ>::Ptr arrow_line_cloud_in_{
      new pcl::PointCloud<pcl::PointXYZ>};
  // 停车线的点云：用于存储停车线的点云数据
  pcl::PointCloud<pcl::PointXYZ>::Ptr slot_cloud_in_{
      new pcl::PointCloud<pcl::PointXYZ>};

  // 上一帧关键帧
  TimedPose pre_key_pose_;  // Pose of last key frame
  // wheel_measurements_: 轮速计测量值的队列，用于存储历史轮速数据
  std::deque<WheelMeasurement> wheel_measurements_;

  // ipm图像到车辆坐标系的变换矩阵
  Eigen::Affine3d T_vehicle_ipm_;
  // 当前帧：存储当前图像帧的处理数据
  Frame curr_frame_;
  // 地图数据类别：存储环境的地图信息
  Map avp_map_;  // avp map data
  PrimitiveSet map_primitives_;
  // 扩展卡尔曼滤波类：用于状态估计和融合
  EKF ekf_;      // EKF state estimation
  // 地图可视化类智能指针：用于显示地图和车辆的位置
  std::shared_ptr<MapViewer> map_viewer_;
  // std::shared_ptr<ViewerInterface> viewer_{nullptr}; // viewer for debug
  AdaptiveStatus adaptive_status_;

  std::string log_directory_;
  std::string semantic_inlier_log_path_;
  std::string residual_log_path_;
  std::string iteration_log_path_;
  std::string hessian_log_path_;
  std::string hessian_iter_log_path_;
  std::string observation_log_path_;
  std::string primitive_plot_dir_;
  std::string map_primitive_export_path_;
  std::string primitive_snapshot_path_;
};
