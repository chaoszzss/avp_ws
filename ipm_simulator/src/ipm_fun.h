#pragma once
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>
// #include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

// IPM融合策略枚举
enum class BlendingStrategy {
  SIMPLE_AVERAGE,     // 简单均值融合（原始方法）
  WEIGHTED_ADVANCED,  // 改进的加权融合（默认）
  WEIGHTED_SIMPLE,    // 简化的加权融合
  HYBRID             // 混合策略：中心用主导相机，边缘用融合
};

// 定义MEI鱼眼相机模型的结构体
struct MEICamera {
  // 构造函数
  MEICamera(const std::string &intrinsicFile, const std::string &extrinsicFile);
  // 析构函数
  ~MEICamera() = default;
  
  // 打印内参文件中的相机参数（图像宽高、MEI模型参数、畸变参数、相机内参）
  // 和外参数（相机坐标系->车辆坐标系旋转四元数、平移向量）
  void DebugString() const;
  
  // 定义相机内参和外参
  bool is_valid_{false};
  std::string id_;
  int height_, width_;
  double xi_;             // MEI相机模型的镜面参数
  cv::Mat K_, D_;         // K：3x3相机内参矩阵（包含焦距和主点）  D_：4x1畸变参数（k1, k2, p1, p2）
  Eigen::Affine3d T_vehicle_cam_;  // Extrinsic: from camera to vehicle（相机到车辆坐标系的变换矩阵（外参））
};

// 管理多个相机并生成IPM图像
class IPM {
 public:
  // 构造函数
  IPM();
  // 析构函数
  ~IPM() = default;

  // 添加相机并加载参数
  void AddCamera(const std::string &intrinsicsFile,
                 const std::string &extrinsicFile);

  // 设置融合策略
  void SetBlendingStrategy(BlendingStrategy strategy);

  // 启用/禁用后处理
  void SetPostProcessingEnabled(bool enabled);

  // 调试：导出权重与主导相机信息
  void SetDebugWeightExport(bool enable);
  void ExportWeightDebugImages(const std::string& directory,
                               const std::string& tag) const;

  cv::Mat GenerateIPMImage(const std::vector<cv::Mat> &images) const;
  cv::Mat GenerateIPMImage(const std::vector<cv::Mat> &images,
                           bool preserve_labels) const;

  // 新增后处理函数声明
  cv::Mat PostProcessLabelIPM(const cv::Mat& ipm_image, const std::vector<cv::Vec3b>& color_map, double max_dist = 50.0) const;

  // 色彩一致性校正函数
  cv::Mat ApplyColorConsistencyCorrection(const cv::Mat& ipm_image) const;

  // 接缝平滑处理函数
  cv::Mat ApplySeamSmoothing(const cv::Mat& ipm_image) const;

 private:
  struct PixelContribution {
    uint16_t camera_idx;
    uint32_t offset;
    float weight;
  };

  struct PixelLookupEntry {
    std::array<PixelContribution, 2> contributions{};
    float inv_weight_sum{0.0f};
    uint8_t count{0};
  };

  cv::Mat GenerateIPMImageImpl(const std::vector<cv::Mat>& images,
                               bool preserve_labels) const;
  void EnsureLookupTable() const;
  void BuildLookupTable() const;

  // 存储多个MEICamera实例的容器
  std::vector<MEICamera> cameras_;

  // IPM图像的尺寸（默认1000x1000）
  int ipm_img_h_ = 1000;
  int ipm_img_w_ = 1000;
  const double pixel_scale_ = 0.02;     // 每个像素对应的实际距离（0.02米/像素）
  const double ipm_img_height_ = 0.0;

  // 融合配置
  BlendingStrategy blending_strategy_ = BlendingStrategy::WEIGHTED_ADVANCED;
  bool post_processing_enabled_ = true;

  mutable std::vector<PixelLookupEntry> pixel_lut_;
  mutable bool lut_dirty_{true};

  // Debug 输出
  bool debug_export_weights_{false};
  mutable bool debug_data_ready_{false};
  mutable std::vector<cv::Mat> debug_weight_maps_;
  mutable cv::Mat debug_dominant_map_;

  // 不同融合策略的实现函数
  cv::Mat ApplySimpleAverageBlending(const std::vector<cv::Mat>& images) const;
  cv::Mat ApplyWeightedAdvancedBlending(const std::vector<cv::Mat>& images) const;
  cv::Mat ApplyWeightedSimpleBlending(const std::vector<cv::Mat>& images) const;
  cv::Mat ApplyHybridBlending(const std::vector<cv::Mat>& images) const;

  void PrepareDebugBuffers() const;
  void ResetDebugBuffers() const;
};

inline void IPM::SetBlendingStrategy(BlendingStrategy strategy) {
  if (blending_strategy_ == strategy) {
    return;
  }
  blending_strategy_ = strategy;
  lut_dirty_ = true;
}

inline void IPM::SetPostProcessingEnabled(bool enabled) {
  post_processing_enabled_ = enabled;
}

inline void IPM::SetDebugWeightExport(bool enable) {
  debug_export_weights_ = enable;
  if (!debug_export_weights_) {
    ResetDebugBuffers();
  }
}

inline void IPM::PrepareDebugBuffers() const {
  if (!debug_export_weights_) {
    return;
  }

  if (debug_weight_maps_.size() != cameras_.size()) {
    debug_weight_maps_.assign(cameras_.size(), cv::Mat());
  }
  for (size_t i = 0; i < cameras_.size(); ++i) {
    cv::Mat& mat = debug_weight_maps_[i];
    if (mat.empty() || mat.rows != ipm_img_h_ || mat.cols != ipm_img_w_ ||
        mat.type() != CV_32FC1) {
      mat = cv::Mat(ipm_img_h_, ipm_img_w_, CV_32FC1, cv::Scalar(0.0f));
    } else {
      mat.setTo(0.0f);
    }
  }

  if (debug_dominant_map_.empty() || debug_dominant_map_.rows != ipm_img_h_ ||
      debug_dominant_map_.cols != ipm_img_w_ ||
      debug_dominant_map_.type() != CV_8UC1) {
    debug_dominant_map_ =
        cv::Mat(ipm_img_h_, ipm_img_w_, CV_8UC1, cv::Scalar(255));
  } else {
    debug_dominant_map_.setTo(255);
  }
  debug_data_ready_ = false;
}

inline void IPM::ResetDebugBuffers() const {
  debug_weight_maps_.clear();
  debug_dominant_map_.release();
  debug_data_ready_ = false;
}

inline void IPM::ExportWeightDebugImages(const std::string& directory,
                                         const std::string& tag) const {
  if (!debug_export_weights_ || !debug_data_ready_) {
    return;
  }

  const std::string dir = directory.empty() ? "." : directory;
  cv::utils::fs::createDirectories(dir);

  for (size_t i = 0; i < debug_weight_maps_.size(); ++i) {
    const cv::Mat& mat = debug_weight_maps_[i];
    if (mat.empty()) {
      continue;
    }

    cv::Mat mat_16u;
    mat.convertTo(mat_16u, CV_16UC1, 65535.0);
    cv::imwrite(dir + "/weight_cam" + std::to_string(i) + "_" + tag + ".png",
                mat_16u);

    cv::Mat mat_8u;
    mat.convertTo(mat_8u, CV_8UC1, 255.0);
    cv::Mat colored;
    cv::applyColorMap(mat_8u, colored, cv::COLORMAP_VIRIDIS);
    cv::imwrite(dir + "/weight_cam" + std::to_string(i) + "_" + tag +
                    "_color.png",
                colored);
  }

  if (!debug_dominant_map_.empty()) {
    cv::Mat lut(1, 256, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::array<cv::Vec3b, 4> colors = {
        cv::Vec3b(255, 0, 0),
        cv::Vec3b(0, 255, 0),
        cv::Vec3b(0, 0, 255),
        cv::Vec3b(255, 255, 0)};
    for (size_t i = 0; i < colors.size(); ++i) {
      lut.at<cv::Vec3b>(static_cast<int>(i)) = colors[i];
    }
    lut.at<cv::Vec3b>(255) = cv::Vec3b(0, 0, 0);

    cv::Mat dominant_color;
    cv::LUT(debug_dominant_map_, lut, dominant_color);
    cv::imwrite(dir + "/dominant_" + tag + ".png", dominant_color);
  }
}
