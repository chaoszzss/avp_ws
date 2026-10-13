//
// Created by ubuntu on 2024-12-31.
// Tong Qin: qintong@sjtu.edu.cn
//

#include "ros/ros.h"
#include "../include/ros_viewer.h"
#include "../include/controller.h"
#include "../include/KinematicModel.h"

// #include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
#include <lgsvl_msgs/VehicleControlData.h>
#include <lgsvl_msgs/CanBusData.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


class Callback {
public:
    Callback(ros::NodeHandle& nh) : 
        nh_(nh), 
        rosviewer(nh),
        // double x, y, psi, v, L, dt;
        vehicle_model(19, 12, M_PI_2, 2, 2, 0.1),
        vehicle_cmd_pub(nh.advertise<lgsvl_msgs::VehicleControlData>("/vehicle_cmd", 1000)),
        last_control_time_(ros::Time::now())  // 新增时间戳记录
    {}

    // 规划的路径回调函数
    void pathcallback(const nav_msgs::PathConstPtr& msg) {
        // 将接收到的路径消息转换为vector<vector<double>>类型
        ROS_INFO("Path frame_id: %s", msg->header.frame_id.c_str());
        path.clear();
        if (msg->poses.empty()) { return;}
        // 第一个点的yaw角无法通过差分计算，初始化为0
        vector<double> prev_point(3);
        prev_point[0] = msg->poses[0].pose.position.x;
        prev_point[1] = msg->poses[0].pose.position.y;
        prev_point[2] = 0; // 初始yaw
        path.push_back(prev_point);

        // 从第二个点开始计算差分yaw
        for (size_t i = 1; i < msg->poses.size(); i++) {
            vector<double> point(3);
            point[0] = msg->poses[i].pose.position.x;
            point[1] = msg->poses[i].pose.position.y;
            
            // 计算当前点与前一点的差分
            double dx = point[0] - path.back()[0];
            double dy = point[1] - path.back()[1];
            point[2] = atan2(dy, dx); // 通过坐标差分计算yaw角
            
            path.push_back(point);
        }
    }

    // 将四元数转换为弧度角，获取yaw角
    template <typename T>
    T GetYaw(const Eigen::Quaternion<T> &rotation) {
    const Eigen::Matrix<T, 3, 1> direction =
        rotation * Eigen::Matrix<T, 3, 1>::UnitX();
    return atan2(direction.y(), direction.x());
    }

    // 合并后的状态回调函数（处理odometry和chassis数据）
    void stateCallback(const nav_msgs::Odometry::ConstPtr& odom_msg,
                        const lgsvl_msgs::CanBusDataConstPtr& chassis_msg) {
        std::lock_guard<std::mutex> lock(model_mutex_);

        // 在stateCallback中添加调试输出
        // ROS_INFO("Odom: x=%.2f y=%.2f | Speed=%.2f m/s, location frame_id: %s",
        // odom_msg->pose.pose.position.x,
        // odom_msg->pose.pose.position.y,
        // chassis_msg->speed_mps,
        // odom_msg->header.frame_id.c_str()
        // );
        
        // 更新位姿
        vehicle_model.x = odom_msg->pose.pose.position.x;
        vehicle_model.y = odom_msg->pose.pose.position.y;
        Eigen::Quaterniond q(
            odom_msg->pose.pose.orientation.w,
            odom_msg->pose.pose.orientation.x,
            odom_msg->pose.pose.orientation.y,
            odom_msg->pose.pose.orientation.z
        );
        vehicle_model.psi = GetYaw(q) - M_PI_2; // 车辆坐标系yaw角（车辆y轴是前进方向）
        
        // 更新速度
        vehicle_model.v = chassis_msg->speed_mps;  // 直接使用底盘速度

        cout << "Vehicle State: x=" << vehicle_model.x << ", y=" << vehicle_model.y
            << ", psi=" << vehicle_model.psi << ", v=" << vehicle_model.v << endl;

        // 更新当前速度（线程安全）
        {
            std::lock_guard<std::mutex> speed_lock(speed_mutex_);
            current_speed = chassis_msg->speed_mps;
        }
        // 判断是否接近终点
        if (!path.empty()) {
            const auto& last_point = path.back();
            double dx = vehicle_model.x - last_point[0];
            double dy = vehicle_model.y - last_point[1];
            double dist_to_end = sqrt(dx*dx + dy*dy);
            ROS_INFO("Distance to end: %.2f", dist_to_end);
            if (dist_to_end < BRAKE_DISTANCE_TH) {
                reached_end = true;
                ROS_WARN("Reaching path end! Braking...");
            }
        }

    }

    // 控制循环（移至主函数循环调用）
    void controlLoop() {
        std::lock_guard<std::mutex> lock(model_mutex_);
        
        // 计算时间步长
        ros::Time current_time = ros::Time::now();
        vehicle_model.dt = (current_time - last_control_time_).toSec();
        last_control_time_ = current_time;
        if (path.empty()) {
            ROS_WARN_THROTTLE(1, "Path is empty");
            return;
        }

        // 构建控制输入
        vector<double> robot_state = {
            vehicle_model.x, 
            vehicle_model.y,
            vehicle_model.psi,
            vehicle_model.v
        };

        ROS_INFO("Robot State: [x: %.2f, y: %.2f, psi: %.2f, v: %.2f]",
            vehicle_model.x, vehicle_model.y, vehicle_model.psi, vehicle_model.v);
        rosviewer.publishPose(vehicle_model.x, vehicle_model.y, vehicle_model.psi);
        
        // ------------------横向控制计算---------------------------
        double lateral_control;
        // lateral_control = controller.PIDController(robot_state, path, vehicle_model);
        // lateral_control = controller.StanlyController(robot_state, path, vehicle_model);
        // lateral_control = controller.LQRController(robot_state, path, vehicle_model);
        lateral_control = controller.PurePursuitController(robot_state, path, vehicle_model);
        // 模型更新
        // double longitudinal_control = 2.0; // constant velocity 2.0m/s
        // vehicle_model.updateState(longitudinal_control, lateral_control);
        ROS_INFO("Lateral Control: %.2f", lateral_control);

        // ------------------纵向控制计算---------------------------
        double acceleration_pct = 0.0;
        double braking_pct = 0.0;
        // 获取当前速度（线程安全）
        double current_speed_local;
        {
            std::lock_guard<std::mutex> speed_lock(speed_mutex_);
            current_speed_local = current_speed;
        }
        // 比例控制加速（可调节系数0.5）
        if (current_speed_local < TARGET_SPEED) {
            double speed_error = TARGET_SPEED - current_speed_local;
            acceleration_pct = std::min(0.15 * speed_error, 1.0); // 系数0.5可调
        } else {
            acceleration_pct = 0.0; // 达到目标速度后油门归零
        }
        // 终点刹车逻辑
        if (reached_end) {
            lgsvl_msgs::VehicleControlData brake_msg;
            braking_pct = 1.0; // 100%刹车力度
            // brake_msg.acceleration_pct = lateral_control; // 维持横向控制 lateral_control
            // brake_msg.braking_pct = 1.0;
            // brake_msg.target_gear = lgsvl_msgs::VehicleControlData::GEAR_PARKING; // 停车档
            // vehicle_cmd_pub.publish(brake_msg);
            ROS_WARN_THROTTLE(1, "Braking...");
            // return; // 仍需退出后续控制
        }
        
        // ------------------发布控制指令----------------------------
        lgsvl_msgs::VehicleControlData control_msg;
        control_msg.header.stamp = ros::Time::now();
        control_msg.acceleration_pct = acceleration_pct;
        control_msg.braking_pct = braking_pct;
        control_msg.target_wheel_angle = lateral_control;
        control_msg.target_gear = lgsvl_msgs::VehicleControlData::GEAR_DRIVE; // 确保档位正确
        // 发布/vehicle_cmd话题
        vehicle_cmd_pub.publish(control_msg);

        // 调试输出
        ROS_INFO("Speed: %.2f/%.2f | Accel: %.2f | Brake: %.2f", 
        current_speed_local, TARGET_SPEED, 
        acceleration_pct, braking_pct);
        
        // 可视化
        // rosviewer.publishPose(vehicle_model.x, vehicle_model.y, vehicle_model.psi);
    }
private:
    ros::NodeHandle nh_;
    RosViewer rosviewer; 
    vector<vector<double>> path;
    KinematicModel vehicle_model;
    Controller controller;
    ros::Timer control_timer;
    std::mutex model_mutex_; // 添加互斥锁
    std::mutex speed_mutex_; // 新增速度锁
    // 控制指令发布者
    ros::Publisher vehicle_cmd_pub;
    ros::Time last_control_time_;  // 新增控制循环时间戳

    // 纵向控制参数
    double TARGET_SPEED = 1.5;       // 目标速度 3 m/s
    double current_speed = 0.0;      // 当前速度
    bool reached_end = false;        // 是否到达终点标志
    double BRAKE_DISTANCE_TH = 2;  // 终点刹车距离阈值（米）
    double speed_error_integral = 0.0;
    double prev_speed_error = 0.0;
};


int main(int argc, char **argv) {
    ros::init(argc, argv, "avp_control_node");
    ros::NodeHandle nh("~");
    Callback callback(nh);
    
    // 消息同步器（同时订阅odometry和chassis）
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/odometry_est", 1);
    message_filters::Subscriber<lgsvl_msgs::CanBusData> chassis_sub(nh, "/chassis", 1);
    typedef message_filters::sync_policies::ApproximateTime<
        nav_msgs::Odometry, 
        lgsvl_msgs::CanBusData
    > SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), odom_sub, chassis_sub);
    sync.registerCallback(boost::bind(&Callback::stateCallback, &callback, _1, _2));
    
    // 路径订阅保持不变
    ros::Subscriber path_sub = nh.subscribe<nav_msgs::Path>(
        "/avp_planning_node/planned_hybridA_path", 
        100, 
        boost::bind(&Callback::pathcallback, &callback, _1)
    );
    
    // 主控制循环
    ros::Rate rate(20); // 10Hz控制频率
    while (ros::ok()) {
        callback.controlLoop();
        rate.sleep();
        ros::spinOnce();  // 保证回调处理
    }
    
    return 0;
}

