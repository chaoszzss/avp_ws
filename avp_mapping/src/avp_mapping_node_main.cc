#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <algorithm>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// 线程
#include <thread>

#include "../include/avp_mapping/avp_mapping.h"
#include "avp_mapping/semantic_downsampler.h"
#include "avp_mapping/slot.h"
#include "../include/avp_mapping/heatmap_utils.h"

geometry_msgs::Point to_ros(const Eigen::Vector3d &pt) {
  geometry_msgs::Point ret;
  ret.x = pt.x();
  ret.y = pt.y();
  ret.z = pt.z();
  return ret;
}

class RosViewer : public ViewerInterface {
 public:
  RosViewer(ros::NodeHandle &n, bool imshow, bool autorun)
      : ViewerInterface(imshow, autorun) {
    pub_path_ = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry_ = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_global_dash_pts_ = n.advertise<sensor_msgs::PointCloud>("global_dash_pts", 1000);
    pub_global_arrow_pts_ = n.advertise<sensor_msgs::PointCloud>("global_arrow_pts", 1000);
    pub_global_slot_pts_ = n.advertise<sensor_msgs::PointCloud>("global_slot_pts", 1000);
    pub_mesh_ = n.advertise<visualization_msgs::Marker>("vehicle", 100, true);
    pub_slot_marker_ = n.advertise<visualization_msgs::Marker>("slot_vector", 100, true);
    pub_slot_marker_ipm_ = n.advertise<visualization_msgs::Marker>("slot_vector_ipm", 100, true);
    // 添加占用标记发布器
    pub_occupancy_marker_ = n.advertise<visualization_msgs::Marker>("occupancy_marker", 1000);
    path_.header.frame_id = "world";
  }
  ~RosViewer(){
    ros::shutdown();
  }

  void displayAvpMap(const TimedPose &pose, const Map &avp_map, const std::vector<Slot> &cur_slots) override {
    // publish pose
    publishPose(pose);
    // publish slot
    publishSlots(avp_map.getAllSlots(), pose.time_, pub_slot_marker_, true);
    publishSlots(cur_slots, pose.time_, pub_slot_marker_ipm_, false);
    // publish semantic points
    auto grid_arrow = avp_map.getSemanticElement(SemanticLabel::kArrowLine);
    publishPoints({grid_arrow.begin(), grid_arrow.end()}, pose.time_, pub_global_arrow_pts_);
    auto grid_dash = avp_map.getSemanticElement(SemanticLabel::kDashLine);
    publishPoints({grid_dash.begin(), grid_dash.end()}, pose.time_, pub_global_dash_pts_);
    // 发布占用标记
    publishOccupancyMarkers(avp_map.getAllSlots(), pose.time_);

    // ros::spinOnce();
  }

 private:
  void publishSlots(const std::vector<Slot> &slots, double time, ros::Publisher &publisher, bool avp_slots){
    if (slots.empty()) {
      return;
    }
    visualization_msgs::Marker line_list;
    line_list.header.frame_id = "world";
    line_list.header.stamp = ros::Time(time);
    line_list.ns = "lines";
    line_list.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;
    line_list.type = visualization_msgs::Marker::LINE_LIST;
    line_list.scale.x = 0.08;  // 线宽 0.1 -> 0.06
    line_list.id = 0;

    // if(avp_slots){
    //   // line_list.color.g = 1.0; 
    //   line_list.color.r = 0.28f; // #47596B 灰蓝
    //   line_list.color.g = 0.35f;
    //   line_list.color.b = 0.42f;
    // }else{
    //   line_list.color.r = 1.0;
    // }
    // line_list.color.a = 1.0;

    if (avp_slots) {
      line_list.color.r = 0.30f; // #4C9FBF
      line_list.color.g = 0.62f;
      line_list.color.b = 0.75f;
    } else {
      line_list.color.r = 0.28f; // #47596B
      line_list.color.g = 0.35f;
      line_list.color.b = 0.42f;
    }
    line_list.color.a = 1.0f;  

    // 遍历每个车位
    for(const auto &slot : slots){
        // 将车位角点转换为ROS格式
        auto p0 = to_ros(slot.corners_[0]);
        auto p1 = to_ros(slot.corners_[1]);
        auto p2 = to_ros(slot.corners_[2]);
        auto p3 = to_ros(slot.corners_[3]);
        // 将车位角点添加到line_list中
        line_list.points.push_back(p0);
        line_list.points.push_back(p1);
        line_list.points.push_back(p1);
        line_list.points.push_back(p2);
        line_list.points.push_back(p2);
        line_list.points.push_back(p3);
        line_list.points.push_back(p3);
        line_list.points.push_back(p0);
    }
    // 发布车位线
    publisher.publish(line_list);
  }

  void publishPoints(const std::vector<Eigen::Vector3i> &grid, double time, ros::Publisher &publisher){
    sensor_msgs::PointCloud global_cloud;
    global_cloud.header.frame_id = "world";
    global_cloud.header.stamp = ros::Time(time);
    for (Eigen::Vector3i pt: grid) {
      geometry_msgs::Point32 p;
      p.x = pt.x() * kPixelScale;
      p.y = pt.y() * kPixelScale;
      p.z = pt.z() * kPixelScale;
      global_cloud.points.push_back(p);
    }
    publisher.publish(global_cloud);
  }

  void publishPose(const TimedPose &pose){
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
    pub_odometry_.publish(odometry);
    
    // pub path
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = "world";
    pose_stamped.header.stamp = ros::Time(pose.time_);
    pose_stamped.pose = odometry.pose.pose;
    path_.poses.push_back(pose_stamped);
    if (path_.poses.size() > 10000) {
      path_.poses.erase(path_.poses.begin());
    }
    pub_path_.publish(path_);
    
    // pub tf
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion tf_q;
    transform.setOrigin(tf::Vector3(position(0),
                                    position(1),
                                    position(2)));
    tf_q.setW(q.w());
    tf_q.setX(q.x());
    tf_q.setY(q.y());
    tf_q.setZ(q.z());
    transform.setRotation(tf_q);
    br.sendTransform(tf::StampedTransform(transform, ros::Time(pose.time_),
                                          "world", "vehicle"));
    
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
    pub_mesh_.publish(meshROS);
  }

  // 发布占用标记
  void publishOccupancyMarkers(const std::vector<Slot> &slots, double time) {
    // 先删除之前的标记
    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = "world";
    delete_marker.header.stamp = ros::Time(time);
    delete_marker.ns = "occupancy";
    delete_marker.action = visualization_msgs::Marker::DELETEALL;
    pub_occupancy_marker_.publish(delete_marker);
    if (slots.empty()) {
      return;
    }

    for (int i = 0; i < slots.size(); ++i) {
      const auto &slot = slots[i];
      if (!slot.occupied) {
        continue;
      }

      Eigen::Vector3d center =
          0.25 * (slot.corners_[0] + slot.corners_[1] + slot.corners_[2] + slot.corners_[3]);
      Eigen::Vector3d edge_long = slot.corners_[1] - slot.corners_[0];
      Eigen::Vector3d edge_short = slot.corners_[3] - slot.corners_[0];
      const double length = edge_long.head<2>().norm();
      const double width = edge_short.head<2>().norm();
      const double yaw = std::atan2(edge_long.y(), edge_long.x());
      tf::Quaternion q = tf::createQuaternionFromYaw(yaw);

      visualization_msgs::Marker fill_marker;
      fill_marker.header.frame_id = "world";
      fill_marker.header.stamp = ros::Time(time);
      fill_marker.ns = "occupancy_fill";
      fill_marker.id = i;
      fill_marker.type = visualization_msgs::Marker::CUBE;
      fill_marker.action = visualization_msgs::Marker::ADD;
      fill_marker.pose.position = to_ros(center);
      fill_marker.pose.orientation.x = q.x();
      fill_marker.pose.orientation.y = q.y();
      fill_marker.pose.orientation.z = q.z();
      fill_marker.pose.orientation.w = q.w();
      fill_marker.scale.x = std::max(0.1, length);
      fill_marker.scale.y = std::max(0.1, width);
      fill_marker.scale.z = 0.1;
      fill_marker.color.r = 1.0;
      fill_marker.color.g = 0.0;
      fill_marker.color.b = 0.0;
      fill_marker.color.a = 0.45;
      pub_occupancy_marker_.publish(fill_marker);
    }
  }

  nav_msgs::Path path_; // trajectory
  ros::Publisher pub_path_, pub_odometry_, pub_mesh_;
  ros::Publisher pub_global_dash_pts_, pub_global_arrow_pts_, pub_global_slot_pts_;
  ros::Publisher pub_slot_marker_, pub_slot_marker_ipm_;
  ros::Publisher pub_occupancy_marker_; // 新增的占用标记发布器
};

void syncCallback(const sensor_msgs::ImageConstPtr& img_msg,
                  const nav_msgs::Odometry::ConstPtr& odom_msg,
                  AvpMapping* mapping) {
    // 添加回调频率控制
    static size_t callback_counter = 0;
    static ros::Time last_callback_time = ros::Time::now();
    
    ros::Time current_time = ros::Time::now();
    if ((current_time - last_callback_time).toSec() < 0.05) { // 20Hz限频
        ROS_WARN_THROTTLE(1, "Callback frequency too high! Dropping messages.");
        return;
    }
    last_callback_time = current_time;

    // 添加时间戳校验
    double time_diff = fabs(img_msg->header.stamp.toSec() - odom_msg->header.stamp.toSec());
    if (time_diff > 0.1) { // 与max_interval_duration一致
      ROS_WARN("Time mismatch: img=%.3f, odom=%.3f (diff=%.3fs)", 
              img_msg->header.stamp.toSec(),
              odom_msg->header.stamp.toSec(),
              time_diff);
      return;
    }

    // 处理位姿
    // 在建图阶段，每一帧的位姿是根据测量到的IMU和轮速计数据使用扩展卡尔曼滤波器进行预测得到的。
    // 这里是直接使用了GPS的Ground_Truth数据作为位姿信息，但是在地下车库环境中需要使用IMU和轮速计数据进行预测。
    TimedPose pose;
    pose.time_ = odom_msg->header.stamp.toSec();
    pose.t_ = Eigen::Vector3d(odom_msg->pose.pose.position.x,
                              odom_msg->pose.pose.position.y,
                              odom_msg->pose.pose.position.z);

    // // 在processPose中添加检查
    Eigen::Quaterniond q(odom_msg->pose.pose.orientation.w,
      odom_msg->pose.pose.orientation.x,
      odom_msg->pose.pose.orientation.y,
      odom_msg->pose.pose.orientation.z);
    if (q.coeffs().hasNaN()) {
      ROS_ERROR("Invalid quaternion in odometry message!");
      return;
    }
    q.normalize(); // 确保四元数归一化
    pose.R_ = q;
    
    // 添加处理超时保护
    // ros::WallTime start = ros::WallTime::now();
    // 处理位姿
    mapping->processPose(pose);
    // ROS_INFO("Pose processed: %f", pose.time_);

    // 处理图像
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
        if (cv_ptr->image.empty()) {
          ROS_ERROR("Received empty image!");
          return;
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    // ros::WallTime start = ros::WallTime::now();
    // std::cout << "Callback counter: " << ++callback_counter << std::endl;
    mapping->processImage(pose.time_, cv_ptr->image);
    // 计算处理时间
    // double elapsed = (ros::WallTime::now() - start).toSec();
    // ROS_INFO("Callback耗时: %.3fs", elapsed);
    // if (elapsed > 0.1) {
    //   ROS_WARN("Callback耗时过长: %.3fs", elapsed);
    // }


    // ROS_INFO("Image processed: %f", pose.time_);
}

AvpMapping mapping;
int main(int argc, char **argv) {
  ros::init(argc, argv, "avp_mapping");
  ros::NodeHandle nh("~");
  bool show_img, auto_run;
  nh.param("show_img", show_img, true);
  nh.param("auto_run", auto_run, true);

  double semantic_voxel_size;
  if (nh.getParam("semantic_voxel_size", semantic_voxel_size)) {
    avp_mapping::SemanticDownsampler::Config cfg;
    cfg.voxel_size = std::max(semantic_voxel_size, kPixelScale);
    avp_mapping::SetSemanticDownsamplerConfig(cfg);
    ROS_INFO_STREAM("Semantic downsampling voxel size set to " << cfg.voxel_size << " m");
  }

  auto viewer = std::make_shared<RosViewer>(nh, show_img, auto_run);
  // 确保mapping对象在回调期间保持有效
  // AvpMapping mapping;
  // 使用智能指针或确保对象生命周期
  // auto mapping_ptr = std::make_shared<AvpMapping>();
  mapping.setViewer(viewer);

  // 使用message_filters同步图像和位姿（订阅话题）
  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/ipm_label", 10);  // 队列大小设置为30
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/gps_odom", 100); // 队列大小设置为150

  // 配置同步策略参数
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry> MySyncPolicy;
  // 创建同步器时直接传入队列大小参数 [3]
  message_filters::Synchronizer<MySyncPolicy> sync(
      MySyncPolicy(30),  // 队列大小通过构造函数设置
      image_sub,
      odom_sub
  );
  // 通过同步器方法设置其他参数 [1]
  sync.setMaxIntervalDuration(ros::Duration(0.2));  // 跨主题时间容差
  // sync.setAgePenalty(0.1);  // 可选参数：时间偏移惩罚系数 [2]
  // 注册回调函数
  sync.registerCallback(boost::bind(&syncCallback, _1, _2, &mapping));
  
  // 输出订阅信息
  ROS_INFO_STREAM("Subscribed to topics: " 
    << image_sub.getTopic() << " and " 
    << odom_sub.getTopic());

  // 记录建图起点（第一个接收到的位姿）
  bool start_pose_saved = false;
  TimedPose start_pose;
  
  // 这种方式错误
  // ros::spin();
  
  // 这种方式错误
  // ros::Rate rate(30); // 控制循环频率
  // while (ros::ok()) {
  //   ros::spinOnce();
  //   rate.sleep();
  // }

  // 正确：应统一使用一种spin方式
  // ros::AsyncSpinner spinner(4); // 推荐异步spinner,使用多个线程处理回调
  ros::AsyncSpinner spinner(std::thread::hardware_concurrency()); // 使用全部CPU核心
  spinner.start();
  ros::waitForShutdown();

  // 节点结束时保存地图和生成热力图
  std::string map_save_dir;
  nh.param<std::string>("map_save_dir", map_save_dir,
                        "/home/chaoszzss/avp_ws/src/avp_mapping/src/AVPMapping/output");
  if (!map_save_dir.empty() && map_save_dir.back() != '/') {
    map_save_dir.push_back('/');
  }
  mapping.getMap().save(map_save_dir + "avp_map_sim.bin");

  std::string semantic_heatmap_path;
  nh.param<std::string>("semantic_heatmap_path", semantic_heatmap_path,
                        map_save_dir + "semantic_coverage_heatmap.png");

  auto semantic_extent =
      avp_mapping::heatmap::ComputeSemanticExtent(mapping.getMap(), 1.0);
  if (semantic_extent.isValid()) {
    auto heatmap =
        avp_mapping::heatmap::RenderSemanticCoverageHeatmap(mapping.getMap(),
                                                            semantic_extent,
                                                            0.5);
    if (!heatmap.visualization.empty()) {
      cv::imwrite(semantic_heatmap_path, heatmap.visualization);
      cv::FileStorage fs(semantic_heatmap_path + ".yml", cv::FileStorage::WRITE);
      fs << "resolution_m" << heatmap.resolution;
      fs << "extent_min_x" << heatmap.extent.min_x;
      fs << "extent_max_x" << heatmap.extent.max_x;
      fs << "extent_min_y" << heatmap.extent.min_y;
      fs << "extent_max_y" << heatmap.extent.max_y;
      fs << "counts" << heatmap.counts;
      fs.release();
      ROS_INFO("Semantic coverage heatmap saved to %s",
               semantic_heatmap_path.c_str());
    }
  }

  ROS_INFO("Semantic map saved to %savp_map_sim.bin", map_save_dir.c_str());

  return 0;
}
