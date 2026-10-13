// //
// // Created by ubuntu on 2024-12-31.
// // Tong Qin: qintong@sjtu.edu.cn
// //

// #include "ros/ros.h"
// #include "../include/ros_viewer.h"
// #include "../include/controller.h"
// #include "../include/KinematicModel.h"

// // #include <std_msgs/Float64MultiArray.h>
// #include <std_msgs/Float64.h>
// #include <lgsvl_msgs/VehicleControlData.h>
// #include <lgsvl_msgs/CanBusData.h>

// #include <message_filters/subscriber.h>
// #include <message_filters/synchronizer.h>
// #include <message_filters/sync_policies/approximate_time.h>

// class Callback {
//     public:
//         Callback(ros::NodeHandle& nh) : 
//             nh_(nh), rosviewer(nh), 
//             control_timer(nh.createTimer(ros::Duration(0.1), &Callback::controlLoop, this)), // 定时器初始化
//             // double x, y, psi, v, L, dt;
//             // startPose.R_ = Eigen::Quaterniond(1, 0, 0, 0);
//             vehicle_model(0, 0, 0, 2, 2, 0.1), // 车辆模型初始化 0, 0, 0, 2, 2, 0.1
//             vehicle_cmd_pub(nh.advertise<lgsvl_msgs::VehicleControlData>("/vehicle_cmd", 1000))  // 指令发布器初始化
//         {}

//         // 规划的路径回调函数
//         void pathcallback(const nav_msgs::PathConstPtr& msg) {
//             // 将接收到的路径消息转换为vector<vector<double>>类型
//             path.clear();
//             if (msg->poses.empty()) { return;}
//             // 第一个点的yaw角无法通过差分计算，初始化为0
//             vector<double> prev_point(3);
//             prev_point[0] = msg->poses[0].pose.position.x;
//             prev_point[1] = msg->poses[0].pose.position.y;
//             prev_point[2] = 0; // 初始yaw
//             path.push_back(prev_point);

//             // 从第二个点开始计算差分yaw
//             for (size_t i = 1; i < msg->poses.size(); i++) {
//                 vector<double> point(3);
//                 point[0] = msg->poses[i].pose.position.x;
//                 point[1] = msg->poses[i].pose.position.y;
                
//                 // 计算当前点与前一点的差分
//                 double dx = point[0] - path.back()[0];
//                 double dy = point[1] - path.back()[1];
//                 point[2] = atan2(dy, dx); // 通过坐标差分计算yaw角
                
//                 path.push_back(point);
//             }
//         }

//         // 将四元数转换为弧度角，获取yaw角
//         template <typename T>
//         T GetYaw(const Eigen::Quaternion<T> &rotation) {
//         const Eigen::Matrix<T, 3, 1> direction =
//             rotation * Eigen::Matrix<T, 3, 1>::UnitX();
//         return atan2(direction.y(), direction.x());
//         }

//         // 车辆位置信息回调函数
//         void posecallback(const nav_msgs::Odometry::ConstPtr& msg) {
//             // 加锁
//             std::lock_guard<std::mutex> lock(model_mutex_);

//             // 获取里程计数据并解析为位置和姿态信息
//             vehicle_model.x = msg->pose.pose.position.x;
//             vehicle_model.y = msg->pose.pose.position.y;
//             Eigen::Quaterniond q(
//             msg->pose.pose.orientation.x,
//             msg->pose.pose.orientation.y,
//             msg->pose.pose.orientation.z,
//             msg->pose.pose.orientation.w
//             );

//             vehicle_model.psi = GetYaw(q); // 获取yaw角
//             // vehicle_model.v = msg->twist.twist.linear.x; // 获取线速度
//             vehicle_model.L = 2;
//         }

//         // 获取车辆实时状态的回调函数
//         void getChassiscallback(const lgsvl_msgs::CanBusDataConstPtr &chassis_speed) {
//             std::lock_guard<std::mutex> lock(speed_mutex_);
//             current_speed = chassis_speed->speed_mps;
//             vehicle_model.v = current_speed; // 获取线速度
//         }

//         // 车辆控制部分
//         void controlLoop(const ros::TimerEvent& event) {
//             // 加锁
//             std::lock_guard<std::mutex> lock(model_mutex_);

//             // 检查路径是否有效
//             if (path.empty()) {
//                 ROS_WARN("Path is empty. Waiting for path message...");
//                 return; // 直接返回，不执行后续控制逻辑
//             }

//             // 车辆模型的时间步长要和定时器的时间步长一致
//             vehicle_model.dt = event.current_real.toSec() - event.last_real.toSec();
//             // vehicle_model.dt = 0.1;
            
//             vector<double> robot_state(4);
//             robot_state[0] = vehicle_model.x;
//             robot_state[1] = vehicle_model.y;
//             robot_state[2] = vehicle_model.psi;
//             robot_state[3] = vehicle_model.v;
    
//             double lateral_control;
//             // lateral_control = controller.PIDController(robot_state, path, vehicle_model);
//             // lateral_control = controller.StanlyController(robot_state, path, vehicle_model);
//             lateral_control = controller.PurePursuitController(robot_state, path, vehicle_model);
//             // lateral_control = controller.LQRController(robot_state, path, vehicle_model);

//             // update state
//             // double longitudinal_control = 2.0; // constant velocity 2.0m/s
//             // vehicle_model.updateState(longitudinal_control, lateral_control);

//             // 发布控制命令
//             /**
//              * # lgsvl_msgs/VehicleControlData.msg
//              * float64 acceleration_pct    # 油门百分比 [0, 1]
//              * float64 braking_pct         # 刹车百分比 [0, 1]
//              * float64 target_wheel_angle  # 目标转向角（单位：弧度）
//              * enum {
//                     GEAR_NEUTRAL = 0u,   // 空挡（无动力传递）
//                     GEAR_DRIVE   = 1u,   // 前进挡（正常驾驶模式）
//                     GEAR_REVERSE = 2u,   // 倒挡
//                     GEAR_PARKING = 3u,   // 停车挡（锁定车轮）
//                     GEAR_LOW     = 4u,   // 低速挡（特殊场景，如爬坡）
//                     };
//              */
//             // 计算纵向控制量（例如：目标速度转换为油门/刹车百分比）
//             double target_speed = 1; // 假设目标速度为2.0 m/s
//             double speed_error;
//             {   
//                 // 加锁
//                 std::lock_guard<std::mutex> lock(speed_mutex_);
//                 speed_error = target_speed - current_speed;
//             }

//             // 根据速度误差设置油门/刹车（示例逻辑）
//             double acceleration_pct = 0.0;
//             double braking_pct = 0.0;
//             if (speed_error > 0) {
//                 acceleration_pct = std::min(speed_error / target_speed, 1.0); // 油门百分比
//             } 

//             // 构造控制消息
//             lgsvl_msgs::VehicleControlData control_msg;
//             control_msg.acceleration_pct = acceleration_pct;     // 油门百分比
//             control_msg.braking_pct = braking_pct;               // 刹车百分比
//             control_msg.target_wheel_angle = lateral_control;    // 转向角直接赋值（单位：弧度）
//             control_msg.target_gear = 1u; // 设置档位（前进档）
//             control_msg.target_wheel_angular_rate = 0.0; // 目标车轮角速度（单位：弧度/秒）
//             // control_msg.target_gear = lgsvl_msgs::VehicleControlData::GEAR_REVERSE;  // 设置档位（倒车档）

            
//             vehicle_cmd_pub.publish(control_msg); // 发布控制指令
    
//             rosviewer.publishPose(vehicle_model.x, vehicle_model.y, vehicle_model.psi);
//         }

//     private:
//         ros::NodeHandle nh_;
//         RosViewer rosviewer; 
//         vector<vector<double>> path;
//         KinematicModel vehicle_model;
//         Controller controller;
//         ros::Timer control_timer;
//         std::mutex model_mutex_; // 添加互斥锁
//         std::mutex speed_mutex_; // 新增速度锁
//         // 控制指令发布者
//         ros::Publisher vehicle_cmd_pub;
//         double current_speed;
// };

// int main(int argc, char **argv) {
//     ros::init(argc, argv, "avp_control_node");
//     ros::NodeHandle nh("~");
//     Callback callback(nh);

//     // 修改为订阅规划器发布的路径
//     ros::Subscriber path_sub = nh.subscribe<nav_msgs::Path>(
//         "/avp_planning_node/planned_hybridA_path", 100, boost::bind(&Callback::pathcallback, &callback, _1)
//     );

//     // 车辆的位置信息由定位模块发布的/odometry话题提供(x,y,yaw)
//     ros::Subscriber pos_sub = nh.subscribe<nav_msgs::Odometry>(
//         "/odometry_est", 100, boost::bind(&Callback::posecallback, &callback, _1)
//     );

//     // 车辆的实时状态由底盘模块发布的/chassis话题提供
//     ros::Subscriber chassis_sub = nh.subscribe<lgsvl_msgs::CanBusData>(
//         "/chassis", 100, boost::bind(&Callback::getChassiscallback, &callback, _1)
//     );

//     ros::spin();
// }