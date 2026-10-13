// geometric_mapping.cc
#include <cmath>
#include <iostream>
#include <vector>

#include "../include/avp_mapping/geometric_mapping.h" // 使用新的头文件
#include "avp_mapping/slot.h" // 仅为了复用里面的常量

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/features2d.hpp>

#include "avp_mapping/heatmap_utils.h"

// --- GeometricMap 实现 ---
GeometricMap::GeometricMap() : map_cloud_(new pcl::PointCloud<pcl::PointXYZ>) {}

void GeometricMap::addPoints(const std::vector<Eigen::Vector3d>& points) {
    if (points.empty()) {
        return;
    }

    // 将新点添加到点云中(这里的点云存储逻辑有点问题？)
    for (const auto& p_eigen : points) {
        map_cloud_->push_back(pcl::PointXYZ(p_eigen.x(), p_eigen.y(), p_eigen.z()));
    }

    // 使用VoxelGrid进行下采样，防止点云过于密集
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(map_cloud_);
    sor.setLeafSize(0.1f, 0.1f, 0.1f); // 体素大小为10cm
    sor.filter(*cloud_filtered);
    
    map_cloud_ = cloud_filtered;
    std::cout << "Map updated. Total points after downsampling: " << map_cloud_->size() << std::endl;
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr GeometricMap::getPointCloud() const {
    return map_cloud_;
}

void GeometricMap::save(const std::string& filename) const {
    if (map_cloud_->empty()) {
        std::cerr << "Point cloud is empty, not saving." << std::endl;
        return;
    }
    pcl::io::savePCDFileASCII(filename, *map_cloud_);
    std::cout << "Geometric map saved to " << filename << " with " << map_cloud_->size() << " points." << std::endl;
}

avp_mapping::heatmap::GeometricHeatmapResult GeometricMap::renderDensityHeatmap(
        double resolution_m,
        double padding_m,
        bool log_scale) const {
    auto extent = avp_mapping::heatmap::ComputeGeometricExtent(map_cloud_, padding_m);
    if (!extent.isValid()) {
        return {};
    }
    return avp_mapping::heatmap::RenderGeometricDensityHeatmap(map_cloud_, extent,
                                                               resolution_m, log_scale);
}


// --- GeometricMapping 实现 ---
GeometricMapping::GeometricMapping() {
    pre_key_pose_.time_ = -1;
    // 初始化ORB特征检测器，每帧最多检测500个特征点
    orb_detector_ = cv::ORB::create(2000); // 400 -> 2000
}

bool GeometricMapping::isKeyFrame(const TimedPose &pose) {
    if ((pose.t_ - pre_key_pose_.t_).norm() > 0.2 || // 位置变化阈值
        std::fabs(GetYaw(pre_key_pose_.R_.inverse() * pose.R_)) > 5. * kToRad || // 角度变化阈值
        pose.time_ - pre_key_pose_.time_ > 30 || pre_key_pose_.time_ < -1) {
        pre_key_pose_ = pose;
        return true;
    }
    return false;
}

void GeometricMapping::processPose(const TimedPose &pose) {
    pose_interpolation_.Push(pose);
}

void GeometricMapping::processImage(double time, const cv::Mat &ipm_img) {
    if (time < pose_interpolation_.EarliestTime()) {
        return;
    }
    TimedPose pose;
    pose.time_ = time;
    if (pose_interpolation_.LookUp(pose) && isKeyFrame(pose)) {
        pose_interpolation_.TrimBefore(time);
        std::cout << "Keyframe processed: t = " << pose.t_.transpose().head(2)
                  << ", yaw = " << GetYaw(pose.R_) * kToDeg << std::endl;

        cv::Mat img_gray;
        cv::cvtColor(ipm_img, img_gray, cv::COLOR_BGR2GRAY);
        
        // 调用新的核心函数
        extractFeaturesAndBuildMap(img_gray, pose);
    }
}

Eigen::Vector3d GeometricMapping::ipmPlane2Global(const TimedPose &T_world_vehicle, const cv::Point2f &ipm_point) {
    // 这个函数的物理意义不变，直接复用
    // IPM像素坐标 -> 车辆中心坐标系 (以图像中心为原点)
    // 图像尺寸为1000x1000, 物理尺寸为20mx20m (0.02m/pixel)
    // x轴向前，y轴向左
    Eigen::Vector3d pt_vehicle_center;
    pt_vehicle_center.x() = -(500.0 - ipm_point.x) * 0.02; // y像素对应车辆x
    pt_vehicle_center.y() = (500.0 - ipm_point.y) * 0.02; // x像素对应车辆y
    pt_vehicle_center.z() = 0.0;

    // 车辆中心坐标系 -> 车辆后轴中心坐标系 (车辆参考点)
    // 假设车辆中心在后轴前1.32m
    Eigen::Vector3d pt_vehicle = pt_vehicle_center;
    pt_vehicle.x() = pt_vehicle_center.x();
    pt_vehicle.y() = pt_vehicle_center.y() + 1.32;
    pt_vehicle.z() = pt_vehicle_center.z();

    // 车辆坐标系 -> 世界坐标系
    Eigen::Vector3d pt_global = T_world_vehicle.R_ * pt_vehicle + T_world_vehicle.t_;
    return pt_global;
          
}

void GeometricMapping::extractFeaturesAndBuildMap(const cv::Mat &img_gray, const TimedPose &T_world_vehicle) {
    std::vector<cv::KeyPoint> keypoints;
    // 使用ORB检测特征点
    orb_detector_->detect(img_gray, keypoints);
    
    std::cout << "Detected " << keypoints.size() << " ORB features." << std::endl;
    if (keypoints.empty()) {
        return;
    }

    // 将所有2D特征点转换到3D世界坐标
    std::vector<Eigen::Vector3d> points_in_world;
    points_in_world.reserve(keypoints.size());
    for (const auto& kp : keypoints) {
        points_in_world.push_back(ipmPlane2Global(T_world_vehicle, kp.pt));
    }

    // 将3D点添加到地图中
    geometric_map_.addPoints(points_in_world);

    // 如果有可视化器，则更新显示
    if (viewer_) {
        // dynamic_cast是安全的做法，确保viewer是我们需要的类型
        auto geometric_viewer = std::dynamic_pointer_cast<GeometricViewerInterface>(viewer_);
        if (geometric_viewer) {
            geometric_viewer->displayMap(T_world_vehicle, geometric_map_);
        }
    }
}
