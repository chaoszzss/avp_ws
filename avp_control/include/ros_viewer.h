//
// Created by ubuntu on 25-3-1.
//
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <vector>
#include <Eigen/Dense>

#ifndef AVP_WS_ROS_VIEWER_H
#define AVP_WS_ROS_VIEWER_H
class RosViewer {
public:
    RosViewer() = default;

    RosViewer(ros::NodeHandle &n);

    void publishPose(const double x, const double y, const double yaw);

    void publishReferencePoint(const double x, const double y);

    void publishTrajectory(const std::vector<std::vector<double>> &trajectory);

    nav_msgs::Path path_, reference_;
    ros::Publisher pub_path_, pub_pose_, pub_mesh_, pub_reference_, pub_ref_pts_ ;
};
#endif //AVP_WS_ROS_VIEWER_H
