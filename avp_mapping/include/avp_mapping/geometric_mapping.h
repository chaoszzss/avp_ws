// geometric_mapping.h
#pragma once

#include <vector>
#include <memory>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// 保持 pose_interpolation.h 的引用
#include "../include/avp_mapping/pose_interpolation.h"
#include "avp_mapping/heatmap_utils.h"

// 前向声明，避免循环包含
class GeometricMap;
class TimedPose;

// 我们需要一个新的ViewerInterface来适应点云地图
class GeometricViewerInterface {
public:
    virtual ~GeometricViewerInterface() = default;
    // 更新可视化，显示当前位姿和全局点云地图
    virtual void displayMap(const TimedPose& pose, const GeometricMap& map) = 0;
};


class GeometricMap {
public:
    GeometricMap();
    void addPoints(const std::vector<Eigen::Vector3d>& points);
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr getPointCloud() const;
    void save(const std::string& filename) const;
    avp_mapping::heatmap::GeometricHeatmapResult renderDensityHeatmap(
        double resolution_m = 0.5, double padding_m = 1.0,
        bool log_scale = true) const;
private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
};


class GeometricMapping {
public:
    GeometricMapping();
    ~GeometricMapping() = default;

    void processPose(const TimedPose &pose);
    void processImage(double time, const cv::Mat &ipm_img);
    const GeometricMap& getMap() const { return geometric_map_; }

    // *** FIX: 修改了这里的参数类型 ***
    void setViewer(std::shared_ptr<GeometricViewerInterface> viewer) { viewer_ = viewer; }

private:
    bool isKeyFrame(const TimedPose &pose);
    Eigen::Vector3d ipmPlane2Global(const TimedPose &T_world_vehicle, const cv::Point2f &ipm_point);
    void extractFeaturesAndBuildMap(const cv::Mat &img_gray, const TimedPose &T_world_vehicle);

    TimedPose pre_key_pose_;
    GeometricMap geometric_map_;
    PoseInterpolation pose_interpolation_;
    cv::Ptr<cv::ORB> orb_detector_;

    // *** FIX: 修改了这里的成员类型 ***
    std::shared_ptr<GeometricViewerInterface> viewer_{nullptr}; 
};
