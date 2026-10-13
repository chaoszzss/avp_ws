//
// Created by ubuntu on 2024-12-31.
// Tong Qin: qintong@sjtu.edu.cn
//

#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/UInt8.h>
#include <geometry_msgs/Vector3.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>  // 添加cv_bridge头文件

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ros/ros.h"
#include "visualization_msgs/Marker.h"
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>

#include "../include/avp_localization.h"
#include "map.h"
#include "../include/map_viewer_ros.h"   // 地图可视化

#include <lgsvl_msgs/CanBusData.h>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// 文件操作
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>

// using namespace Eigen;
using namespace std;

ros::Publisher pub_estimation_path, pub_gt_path, pub_measurement_path,
    pub_odometry;
nav_msgs::Path path_estimation, path_gt, path_measurment;
ros::Publisher meshPub;

struct Node {

  Node() {
    // Avoid std::make_shared because AvpLocalization uses aligned Eigen types.
    avp_localization_.reset(new AvpLocalization(DATASET_PATH "avp_map_sim_compressed.bin"));
    map_viewer_ = std::make_shared<RosViewer>(nh_);

    // 发布静态地图数据
    const Map &map = avp_localization_->getMap();
    map_viewer_->displayAvpMap(map, ros::Time::now().toSec());

    // 发布的RVIZ可视化话题
    pub_odometry = nh_.advertise<nav_msgs::Odometry>("/odometry_est", 2000);
    pub_estimation_path = nh_.advertise<nav_msgs::Path>("/estimation_path", 2000);
    pub_gt_path = nh_.advertise<nav_msgs::Path>("/gps_path", 5000);
    meshPub = nh_.advertise<visualization_msgs::Marker>("/vehicle_mesh", 100, true);
    dma_marker_pub_ =
        nh_.advertise<visualization_msgs::Marker>("/dma_ekf/marker", 10);
    dma_state_pub_ = nh_.advertise<std_msgs::UInt8>("/dma_ekf/mode", 50);
    dma_metric_pub_ =
        nh_.advertise<geometry_msgs::Vector3>("/dma_ekf/phi_gamma_quality", 50);

    path_gt.header.frame_id = "world";
    path_estimation.header.frame_id = "world";

    // 初始化数据保存文件
    initializeDataFiles();
  }

  ~Node() {
    // 关闭文件输出流
    if (gps_file_.is_open()) {
      gps_file_.close();
    }
    if (estimation_file_.is_open()) {
      estimation_file_.close();
    }
    std::cout << "Data files saved successfully." << std::endl;
  }

  // 发布估计路径话题
  void publishPath(const TimedPose &pose) {
    Eigen::Vector3d position = pose.t_;
    Eigen::Quaterniond q = pose.R_;

    // pub odometry
    nav_msgs::Odometry odometry;
    odometry.header.frame_id = "world";
    odometry.header.stamp = ros::Time(pose.time_);
    odometry.pose.pose.position.x = position(0);
    odometry.pose.pose.position.y = position(1);
    odometry.pose.pose.position.z = position(2);
    odometry.pose.pose.orientation.x = q.x();
    odometry.pose.pose.orientation.y = q.y();
    odometry.pose.pose.orientation.z = q.z();
    odometry.pose.pose.orientation.w = q.w();
    pub_odometry.publish(odometry);

    // pub path
    geometry_msgs::PoseStamped poseStamped;
    poseStamped.header.frame_id = "world";
    poseStamped.header.stamp = ros::Time(pose.time_);
    poseStamped.pose = odometry.pose.pose;
    path_estimation.poses.push_back(poseStamped);
    if (path_estimation.poses.size() > 50000) {
      path_estimation.poses.erase(path_estimation.poses.begin());
    }
    pub_estimation_path.publish(path_estimation);

    // pub tf
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q_tf; 
    transform.setOrigin(tf::Vector3(position(0), position(1), position(2)));
    q_tf.setX(q.x());
    q_tf.setY(q.y());
    q_tf.setZ(q.z());
    q_tf.setW(q.w());
    transform.setRotation(q_tf);
    br.sendTransform(tf::StampedTransform(transform, ros::Time(pose.time_), 
                                          "world", "map"));
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map"; //?

    // pub Mesh model
    visualization_msgs::Marker meshROS;
    meshROS.header.frame_id = std::string("world");
    meshROS.header.stamp = ros::Time(pose.time_);
    meshROS.ns = "mesh";
    meshROS.id = 0;
    meshROS.type = visualization_msgs::Marker::MESH_RESOURCE;
    meshROS.action = visualization_msgs::Marker::ADD;
    Eigen::Matrix3d frontleftup2rightfrontup;
    frontleftup2rightfrontup << 0, 1, 0, -1, 0, 0, 0, 0, 1;
    Eigen::Matrix3d rot_mesh;
    rot_mesh << -1, 0, 0, 0, 0, 1, 0, 1, 0;
    Eigen::Quaterniond q_mesh;
    q_mesh = q * frontleftup2rightfrontup.transpose() * rot_mesh;
    Eigen::Vector3d t_mesh = q * Eigen::Vector3d(0, 1.5, 0) + position;
    meshROS.pose.orientation.w = q_mesh.w();
    meshROS.pose.orientation.x = q_mesh.x();
    meshROS.pose.orientation.y = q_mesh.y();
    meshROS.pose.orientation.z = q_mesh.z();
    meshROS.pose.position.x = t_mesh(0);
    meshROS.pose.position.y = t_mesh(1);
    meshROS.pose.position.z = t_mesh(2);
    meshROS.scale.x = 1.0;
    meshROS.scale.y = 1.0;
    meshROS.scale.z = 1.0;
    meshROS.color.a = 1.0;
    meshROS.color.r = 1.0;
    meshROS.color.g = 0.0;
    meshROS.color.b = 0.0;
    std::string mesh_resource("package://avp_mapping/launch/car.dae");
    // 确认文件路径是否正确
    // if (!ros::package::getPath("avp_mapping").empty()) {
    //     mesh_resource = "package://avp_mapping/launch/car.dae";
    // } else {
    //     ROS_ERROR("Failed to find car model!");
    // }
    meshROS.mesh_resource = mesh_resource;
    meshPub.publish(meshROS);

    // 保存估计位姿数据到文件
    if (estimation_file_.is_open()) {
      double yaw = GetYaw(q) * kToDeg;  // 转换为度数
      estimation_file_ << pose.time_ << " "
                      << position(0) << " " << position(1) << " " << position(2) << " "
                      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
                      << yaw << std::endl;
    }
  }
  

  void addOdomGps(const nav_msgs::OdometryConstPtr &msg) {

    Eigen::Vector3d t;
    Eigen::Quaterniond q;
    t.x() = msg->pose.pose.position.x;
    t.y() = msg->pose.pose.position.y;
    t.z() = msg->pose.pose.position.z;
    q.x() = msg->pose.pose.orientation.x;
    q.y() = msg->pose.pose.orientation.y;
    q.z() = msg->pose.pose.orientation.z;
    q.w() = msg->pose.pose.orientation.w;

    // cout << "gps: t = " << t.x() << ", " << t.y() << ", " << t.z() << endl;
    // cout << "gps: yaw = " <<  GetYaw(q) << endl;

    if (!ekf_inited_) { // set ref pose
      ekf_inited_ = true;
      // add errors to init
      // avp_localization_->initState(msg->header.stamp.toSec(), t.x() - 1., t.y() + 1.,
      //                              GetYaw(q) + 5. * kToRad);
      // 不给初始状态添加误差
      avp_localization_->initState(msg->header.stamp.toSec(), t.x(), t.y(),
      GetYaw(q));
    }
    // printf("gps_gt: t = %.4f, x= %.3f , y = %.3f, yaw = %.2f \n",
    //        msg->header.stamp.toSec(), t.x(), t.y(), kToDeg * GetYaw(q));

    // 将当前GPS位姿添加到路径并发布
    geometry_msgs::PoseStamped poseStamped_gt;
    poseStamped_gt.header = msg->header;
    poseStamped_gt.header.frame_id = "world";
    poseStamped_gt.pose = msg->pose.pose;
    path_gt.poses.push_back(poseStamped_gt);
    if (path_gt.poses.size() > 50000) {
      path_gt.poses.erase(path_gt.poses.begin());
    }
    pub_gt_path.publish(path_gt); // 直接发布path_gt

    // 保存GPS数据到文件
    if (gps_file_.is_open()) {
      double yaw = GetYaw(q) * kToDeg;  // 转换为度数
      gps_file_ << msg->header.stamp.toSec() << " "
                << t.x() << " " << t.y() << " " << t.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
                << yaw << std::endl;
    }
  }

  // void addImuSpeed(const sensor_msgs::ImuConstPtr &imu,
  //                  const geometry_msgs::Vector3StampedConstPtr &speed) {
  //   if (ekf_inited_) {
  //     WheelMeasurement wheel_measurement;
  //     wheel_measurement.time_ = imu->header.stamp.toSec();
  //     wheel_measurement.velocity_ = speed->vector.y;
  //     wheel_measurement.yaw_rate_ = imu->angular_velocity.z;
  //     avp_localization_->processWheelMeasurement(wheel_measurement);
  //   }
  // }


  void addImuChassis(const lgsvl_msgs::CanBusDataConstPtr &chassis_speed, 
      const sensor_msgs::ImuConstPtr &imu) {
    if (ekf_inited_) {
      WheelMeasurement wheel_measurement;
      wheel_measurement.time_ = chassis_speed->header.stamp.toSec();
      double vx = chassis_speed->linear_velocities.x;
      double vz = chassis_speed->linear_velocities.z;
      // 档位信息判断是前进还是后退
      int gear = chassis_speed->selected_gear;

      // wheel_measurement.velocity_ = sqrt(vx * vx + vz * vz);  // 速度的数值计算是正确的
      if (gear == 1) {
        // wheel_measurement.velocity_ = chassis_speed->speed_mps; // 前进
        wheel_measurement.velocity_ = sqrt(vx * vx + vz * vz); // 前进
      } else if (gear == 2) {
        // wheel_measurement.velocity_ = -chassis_speed->speed_mps; // 后退
        wheel_measurement.velocity_ = -sqrt(vx * vx + vz * vz); // 后退
      }
      wheel_measurement.yaw_rate_ = imu->angular_velocity.z;

      // 估算转向角 (如果没有直接测量，使用偏航率和速度估算)
      if (std::abs(wheel_measurement.velocity_) > 0.1) {
        // 使用自行车模型估算转向角: tan(δ) = L * ω / v
        wheel_measurement.steering_angle_ = std::atan(WHEELBASE * imu->angular_velocity.z /
                                                      std::abs(wheel_measurement.velocity_));
        // 限制转向角在合理范围内
        wheel_measurement.steering_angle_ = std::max(-MAX_STEERING_ANGLE,
                                                   std::min(MAX_STEERING_ANGLE,
                                                          wheel_measurement.steering_angle_));
      } else {
        wheel_measurement.steering_angle_ = 0.0;
      }

      avp_localization_->processWheelMeasurement(wheel_measurement);
    }
  }
  // //CompressedImage 消息的回调函数
  // void addIPmImage(const sensor_msgs::ImageConstPtr &msg) {
  //   cv::Mat image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
  //   if (!image.empty()) {
  //     double cur_time = msg->header.stamp.toSec();
  //     avp_localization_->processImage(cur_time, image);

  //     // 发布估计路径
  //     // 获取当前位姿
  //     TimedPose current_pose = avp_localization_->getCurrentPose();
  //     // 发布位姿
  //     publishPath(current_pose);
  //   }
  // }

  //CompressedImage 消息的回调函数
  void addIPmImage(const sensor_msgs::CompressedImageConstPtr &msg) {
    cv::Mat image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (!image.empty()) {
      double cur_time = msg->header.stamp.toSec();
      avp_localization_->processImage(cur_time, image);

      // 发布估计路径
      // 获取当前位姿
      TimedPose current_pose = avp_localization_->getCurrentPose();
      // 发布位姿
      publishPath(current_pose);
      publishAdaptiveDiagnostics(current_pose);
    }
  }

  bool ekf_inited_{false};
  std::shared_ptr<AvpLocalization> avp_localization_;
  std::shared_ptr<RosViewer> map_viewer_;

  ros::NodeHandle nh_;
  ros::Publisher pub_estimation_path, pub_gt_path, pub_measurement_path,
                 pub_odometry;
  nav_msgs::Path path_estimation, path_gt, path_measurment;
  ros::Publisher meshPub;
  ros::Publisher dma_marker_pub_;
  ros::Publisher dma_state_pub_;
  ros::Publisher dma_metric_pub_;

  // 新增成员变量
  ros::Publisher pub_map_;

private:
  // 数据保存文件流
  std::ofstream gps_file_;
  std::ofstream estimation_file_;

  // 初始化数据保存文件
  void initializeDataFiles() {
    // 获取当前时间作为文件名后缀
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    // 打开GPS数据文件
    std::string gps_filename = DATASET_PATH + std::string("../output/gps_data_") + timestamp + ".txt";
    gps_file_.open(gps_filename);
    if (gps_file_.is_open()) {
      gps_file_ << "# GPS Data - Timestamp(s) X(m) Y(m) Z(m) Qx Qy Qz Qw Yaw(deg)" << std::endl;
      gps_file_ << std::fixed << std::setprecision(6);
      std::cout << "GPS data file opened: " << gps_filename << std::endl;
    } else {
      std::cerr << "Failed to open GPS data file: " << gps_filename << std::endl;
    }

    // 打开估计位姿数据文件
    std::string estimation_filename = DATASET_PATH + std::string("../output/estimation_data_") + timestamp + ".txt";
    estimation_file_.open(estimation_filename);
    if (estimation_file_.is_open()) {
      estimation_file_ << "# Estimation Data - Timestamp(s) X(m) Y(m) Z(m) Qx Qy Qz Qw Yaw(deg)" << std::endl;
      estimation_file_ << std::fixed << std::setprecision(6);
      std::cout << "Estimation data file opened: " << estimation_filename << std::endl;
    } else {
      std::cerr << "Failed to open estimation data file: " << estimation_filename << std::endl;
    }
  }

  void publishAdaptiveDiagnostics(const TimedPose &pose) {
    AdaptiveStatus status = avp_localization_->getAdaptiveStatus();
    std_msgs::UInt8 state_msg;
    state_msg.data = static_cast<uint8_t>(status.mode);
    dma_state_pub_.publish(state_msg);

    geometry_msgs::Vector3 metric_msg;
    metric_msg.x = status.phi;
    metric_msg.y = status.gamma;
    metric_msg.z = status.quality;
    dma_metric_pub_.publish(metric_msg);

    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp =
        ros::Time(status.stamp > 0.0 ? status.stamp : pose.time_);
    marker.ns = "dma_ekf";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = pose.t_(0);
    marker.pose.position.y = pose.t_(1);
    marker.pose.position.z = pose.t_(2) + 1.5;
    marker.pose.orientation.w = 1.0;
    marker.scale.z = 0.6;
    marker.color.a = 1.0;
    marker.lifetime = ros::Duration(1.0); //刷新频率

    auto setColor = [&](float r, float g, float b) {
      marker.color.r = r;
      marker.color.g = g;
      marker.color.b = b;
    };

    std::string mode_label = "Idle";
    switch (status.mode) {
      case AdaptiveMode::kIdle:
        setColor(0.6f, 0.6f, 0.6f);
        mode_label = "NoMeasurement";
        break;
      case AdaptiveMode::kNormal:
        setColor(0.0f, 0.8f, 0.2f);
        mode_label = "Normal";
        break;
      case AdaptiveMode::kStrongTracking:
        setColor(1.0f, 0.75f, 0.0f);
        mode_label = "StrongTracking";
        break;
      case AdaptiveMode::kObservationDownweight:
        setColor(0.0f, 0.5f, 1.0f);
        mode_label = "ObsDownweight";
        break;
      case AdaptiveMode::kCombined:
        setColor(1.0f, 0.2f, 0.2f);
        mode_label = "Combined";
        break;
    }

    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss.precision(3);
    oss << "DMA-EKF: " << mode_label << "\nphi=" << status.phi
        << "  gamma=" << status.gamma << "\nQ=" << status.quality;
    oss.precision(3);
    oss << "  rms=" << status.fitness << "m";
    oss.precision(1);
    oss << "\ninliers=" << status.inlier_ratio * 100.0 << "%";
    marker.text = oss.str();

    dma_marker_pub_.publish(marker);
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "avp_localization_node");
  ros::NodeHandle nh("~");

  Node node;

  message_filters::Subscriber<sensor_msgs::Imu> sub0(nh, "/imu", 100);
  message_filters::Subscriber<lgsvl_msgs::CanBusData> sub1(nh, "/chassis", 100);

  typedef message_filters::sync_policies::ExactTime<
      sensor_msgs::Imu, lgsvl_msgs::CanBusData>
      sync_pol;
  message_filters::Synchronizer<sync_pol> sync(sync_pol(100), sub0, sub1);
  sync.registerCallback(boost::bind(&Node::addImuChassis, &node, _2, _1));

  ros::Subscriber gps_sub = nh.subscribe<nav_msgs::Odometry>(
      "/gps_odom", 100, boost::bind(&Node::addOdomGps, &node, _1));
  ros::Subscriber ipm_sub = nh.subscribe<sensor_msgs::CompressedImage>(
      "/ipm_semantic_label/compressed", 10, boost::bind(&Node::addIPmImage, &node, _1));

  ros::spin();
}
