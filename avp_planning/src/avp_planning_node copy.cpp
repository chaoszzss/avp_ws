// //
// // Created by ubuntu on 2024-12-31.
// // Tong Qin: qintong@sjtu.edu.cn
// //

// #include <geometry_msgs/Vector3Stamped.h>
// #include <nav_msgs/Odometry.h>
// #include <nav_msgs/Path.h>

// #include <opencv2/opencv.hpp>

// #include <Eigen/Core>
// #include <Eigen/Geometry>

// #include "ros/ros.h"
// #include "visualization_msgs/Marker.h"
// #include <geometry_msgs/PoseStamped.h>
// #include <nav_msgs/Odometry.h>
// #include <nav_msgs/Path.h>
// #include <tf/transform_broadcaster.h>
// #include "../include/avp_map.h"
// #include "../include/ros_viewer.h"
// #include "../include/planner.h"

// #include <csignal>

// // volatile bool g_shutdown = false;
// // // 信号处理函数
// // void sigintHandler(int sig) {
// //     g_shutdown = true;
// //     ros::shutdown();
// // }

// bool newGoalPoint = false;
// Eigen::Vector2d goal(0, 0);
// void navGoalCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
// {
//     ROS_INFO("Received 2D Nav Goal:");
//     ROS_INFO("Position: [x: %.2f, y: %.2f, z: %.2f]",
//              msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
//     ROS_INFO("Orientation: [x: %.2f, y: %.2f, z: %.2f, w: %.2f]",
//              msg->pose.orientation.x, msg->pose.orientation.y,
//              msg->pose.orientation.z, msg->pose.orientation.w);

//     newGoalPoint = true;
//     goal.x() = msg->pose.position.x;
//     goal.y() = msg->pose.position.y;
// }

// // 新增全局变量或类成员（建议封装到类中）
// std::mutex pose_mutex_;
// Eigen::Vector3d current_position(0, 0, 0); // 初始位置
// double current_yaw = 0.0;
// // 定位模块发布的/odometry话题回调函数
// void posecallback(const nav_msgs::OdometryConstPtr &msg) {
//     std::lock_guard<std::mutex> lock(pose_mutex_);
//     // 更新位置
//     current_position.x() = msg->pose.pose.position.x;
//     current_position.y() = msg->pose.pose.position.y;
//     current_position.z() = msg->pose.pose.position.z;
//     // 更新航向角（yaw）
//     Eigen::Quaterniond q(
//         msg->pose.pose.orientation.x,
//         msg->pose.pose.orientation.y,
//         msg->pose.pose.orientation.z,
//         msg->pose.pose.orientation.w
//     );
//     current_yaw = GetYaw(q);
//     // Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0); // ZYX顺序，获取yaw
//     // current_yaw = euler[0];
// }


// int main(int argc, char **argv) {
//     // 注册信号处理函数
//     // signal(SIGINT, sigintHandler);  // 处理Ctrl+C,防止程序资源泄漏

//     ros::init(argc, argv, "avp_planning_node");
//     ros::NodeHandle nh("~");
//     ros::Subscriber sub1 = nh.subscribe("/move_base_simple/goal", 10, navGoalCallback);

//     // 订阅定位模块发布的/odometry话题
//     ros::Subscriber pos_sub = nh.subscribe<nav_msgs::Odometry>(
//         "/odometry", 1000, boost::bind(posecallback, _1)
//     );
//     // ros::topic::waitForMessage<nav_msgs::Odometry>("/odometry", nh);

//     AvpMap avp_map;
//     avp_map.load(DATASET_PATH "avp_map_sim.bin");
//     RosViewer rosViewer(nh);
//     // visualize avp map
//     rosViewer.displayAvpMap(avp_map, 0);

//     // visualize vehicle
//     TimedPose startPose;
//     startPose.time_ = ros::Time::now().toSec();     // 当前时间
//     // std::lock_guard<std::mutex> lock(pose_mutex_);
//     // startPose.t_ = current_position;
//     // startPose.R_ = Eigen::Quaterniond(Eigen::AngleAxisd(current_yaw, Eigen::Vector3d::UnitZ()));
//     startPose.t_= Eigen::Vector3d(19, 16, 0);
//     startPose.R_ = Eigen::Quaterniond(1, 0, 0, 0);
//     // startPose.t_= Eigen::Vector3d(19, 16, 0);   // 0, -3, 0, 2, 2, 0.1
//     // startPose.R_ = Eigen::Quaterniond(1, 0, 0, 0);
//     rosViewer.publishPose(startPose);

//     // get obstacles
//     std::vector<Eigen::Vector3d> obstaclePoints;
//     for (const auto &p : avp_map.getSemanticElement((SemanticLabel::kSlot))) {
//         Eigen::Vector3d tmp_p(p.x() * kPixelScale,
//                 p.y() * kPixelScale,
//                 p.z() * kPixelScale);
//         obstaclePoints.push_back(tmp_p);
//     }

//     // convert start/end/obstaclePts in 2D grid coordinate with resolution 0.5m*0.5m
//     double resolution = 0.5;
//     Vector2i start_index(round(startPose.t_(0) / resolution),
//                    round(startPose.t_(1) / resolution));

//     // 这里的障碍物是将像素坐标转换为分辨率为0.5米的栅格坐标
//     // 所以在后面将节点和障碍物进行比较时，需要将节点坐标也转换为栅格坐标
//     unordered_set<Vector2i, Vector2iHash> obstacles_index;
//     for (auto& p : obstaclePoints) {
//         Vector2i obs_p(round(p.x() / resolution),
//                      round(p.y() / resolution));
//         obstacles_index.insert(obs_p);
//     }

//     ros::Rate loop_rate(10);
//     while(ros::ok()) {
//         loop_rate.sleep();
//         ros::spinOnce();

//         if(!newGoalPoint)
//             continue;
        
//         newGoalPoint = false; // 重置标志位
        
//         // 获取最新起始点
//         Eigen::Vector3d start_pos;
//         double start_yaw;
        
//         // std::lock_guard<std::mutex> lock(pose_mutex_);
//         // start_pos = current_position;
//         // start_yaw = current_yaw;
//         start_pos = startPose.t_;
//         start_yaw = GetYaw(startPose.R_);
        
//         // 更新 Hybrid A* 的起始状态
//         // State startState(start_pos.x(), start_pos.y(), start_yaw, 0, 0, nullptr);

//         Planner planner;

//         // 车辆参数 for hybrid A
//         double wheelbase = 2.5;   // 轴距
//         double step_size = 1.5;   // 步长
//         double max_steer = M_PI / 6; // 最大转向角（45度）
//         // State startState(startPose.t_(0), startPose.t_(1), M_PI / 2, 0, 0, nullptr);
//         State startState(start_pos.x(), start_pos.y(), start_yaw, 0, 0, nullptr);   // 动态更新起始点
//         State goalState(goal.x(), goal.y(), 0, 0, 0, nullptr);
//         std::vector<State*> hybridApath = planner.HybridAStar(startState, goalState,
//                                                                obstacles_index,
//                                                                 wheelbase, step_size, max_steer);
//         // publish result
//         if (hybridApath.empty()) {
//             ROS_WARN("Fail to find Hybrid A* path");
//         } else {
//             std::vector<Eigen::Vector2d> path_points;
//             for (auto &pState : hybridApath) {
//                 Eigen::Vector2d tmp (pState->x, pState->y);
//                 path_points.emplace_back(tmp);
//                 delete pState;
//             }
//             rosViewer.publishHybridATrajectory(path_points);
//         }
//     }

//     ros::spin();
// }
