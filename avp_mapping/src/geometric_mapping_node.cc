// geometric_mapping_node.cc
#include <iostream>
#include <memory>
#include <thread>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <tf_conversions/tf_eigen.h>

#include "../include/avp_mapping/geometric_mapping.h" 
#include "avp_mapping/heatmap_utils.h"

// RosViewer类保持不变
class RosViewer : public GeometricViewerInterface {
public:
    RosViewer(ros::NodeHandle &n) : nh_(n) {
        pub_path_ = n.advertise<nav_msgs::Path>("geometric_path", 10);
        pub_odometry_ = n.advertise<nav_msgs::Odometry>("geometric_odometry", 10);
        pub_mesh_ = n.advertise<visualization_msgs::Marker>("geometric_vehicle", 10, true);
        pub_geometric_map_ = n.advertise<sensor_msgs::PointCloud2>("geometric_map", 1, true);
        path_.header.frame_id = "world";
    }
    ~RosViewer() { ros::shutdown(); }

    void displayMap(const TimedPose& pose, const GeometricMap& map) override {
        publishPose(pose);
        auto cloud = map.getPointCloud();
        if (cloud && !cloud->empty()) {
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*cloud, cloud_msg);
            cloud_msg.header.stamp = ros::Time(pose.time_);
            cloud_msg.header.frame_id = "world";
            pub_geometric_map_.publish(cloud_msg);
        }
    }

private:
   // *** FIX: 修改这个函数 ***
    void publishPose(const TimedPose &pose){
        Eigen::Vector3d position = pose.t_;
        Eigen::Quaterniond q = pose.R_;

        // --- 发布 odometry (这部分不变) ---
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
        
        // --- 发布 path (这部分不变) ---
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = odometry.header;
        pose_stamped.pose = odometry.pose.pose;
        path_.poses.push_back(pose_stamped);
        pub_path_.publish(path_);
        
        // --- 发布 tf (这部分需要修正) ---
        static tf::TransformBroadcaster br;
        tf::Transform transform;

        // 1. 创建临时的tf类型变量(左值)
        tf::Quaternion tf_q;
        tf::Vector3 tf_v;

        // 2. 调用转换函数，将Eigen数据填充到tf变量中
        tf::quaternionEigenToTF(q, tf_q);
        tf::vectorEigenToTF(position, tf_v); // 使用正确的函数名 vectorEigenToTF

        // 3. 使用填充好的tf变量来设置transform
        transform.setRotation(tf_q);
        transform.setOrigin(tf_v);

        // 4. 发布变换
        br.sendTransform(tf::StampedTransform(transform, ros::Time(pose.time_), "world", "vehicle"));
    }

    ros::NodeHandle &nh_;
    nav_msgs::Path path_;
    ros::Publisher pub_path_, pub_odometry_, pub_mesh_;
    ros::Publisher pub_geometric_map_;
};

// ... syncCallback 和 main 函数保持不变 ...
void syncCallback(const sensor_msgs::ImageConstPtr& img_msg,
                  const nav_msgs::Odometry::ConstPtr& odom_msg,
                  GeometricMapping* mapping) {
    double time_diff = fabs(img_msg->header.stamp.toSec() - odom_msg->header.stamp.toSec());
    if (time_diff > 0.1) {
        ROS_WARN("Time difference between image and odom is too large: %.3fs", time_diff);
        return;
    }

    TimedPose pose;
    pose.time_ = odom_msg->header.stamp.toSec();
    pose.t_ = Eigen::Vector3d(odom_msg->pose.pose.position.x,
                              odom_msg->pose.pose.position.y,
                              odom_msg->pose.pose.position.z);
    Eigen::Quaterniond q(odom_msg->pose.pose.orientation.w,
                         odom_msg->pose.pose.orientation.x,
                         odom_msg->pose.pose.orientation.y,
                         odom_msg->pose.pose.orientation.z); 
    pose.R_ = q.normalized() ;
    mapping->processPose(pose);

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    mapping->processImage(pose.time_, cv_ptr->image);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "geometric_mapping_node");
    ros::NodeHandle nh("~");

    GeometricMapping mapping;
    auto viewer = std::make_shared<RosViewer>(nh);
    
    mapping.setViewer(viewer);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/ipm_semantic_label", 10); // ipm_raw
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/gps_odom", 100);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(30), image_sub, odom_sub);
    sync.setMaxIntervalDuration(ros::Duration(0.2));
    sync.registerCallback(boost::bind(&syncCallback, _1, _2, &mapping));

    ROS_INFO("Geometric mapping node started. Subscribing to %s and %s.", 
             image_sub.getTopic().c_str(), odom_sub.getTopic().c_str());

    ros::AsyncSpinner spinner(std::thread::hardware_concurrency());
    spinner.start();
    ros::waitForShutdown();

    std::string map_path;
    nh.param<std::string>("map_save_path", map_path, "/home/chaoszzss/avp_ws/src/avp_mapping/src/AVPMapping/output/geometric_map.pcd");
    mapping.getMap().save(map_path);

    std::string density_heatmap_path;
    nh.param<std::string>("density_heatmap_path", density_heatmap_path,
                          "/home/chaoszzss/avp_ws/src/avp_mapping/src/AVPMapping/output/geometric_density_heatmap.png");

    auto cloud = mapping.getMap().getPointCloud();
    auto extent = avp_mapping::heatmap::ComputeGeometricExtent(cloud, 1.0);
    if (extent.isValid()) {
        auto heatmap = avp_mapping::heatmap::RenderGeometricDensityHeatmap(
            cloud, extent, 0.5);
        if (!heatmap.visualization.empty()) {
            cv::imwrite(density_heatmap_path, heatmap.visualization);

            const std::string raw_path = density_heatmap_path + ".yml";
            cv::FileStorage fs(raw_path, cv::FileStorage::WRITE);
            fs << "resolution_m" << heatmap.resolution;
            fs << "extent_min_x" << heatmap.extent.min_x;
            fs << "extent_max_x" << heatmap.extent.max_x;
            fs << "extent_min_y" << heatmap.extent.min_y;
            fs << "extent_max_y" << heatmap.extent.max_y;
            fs << "density" << heatmap.density;
            fs.release();
            ROS_INFO("Geometric density heatmap saved to %s", density_heatmap_path.c_str());
        }
    }

    return 0;
}
