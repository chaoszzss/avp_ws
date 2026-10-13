#include "ipm_fun.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <thread>
#include <omp.h>
#include <opencv2/core/utils/filesystem.hpp>

// MEICamera结构体的构造函数功能实现
// 从YAML文件读取内参（如畸变参数、投影参数）和外参（相机到车辆的变换矩阵）
MEICamera::MEICamera(const std::string& intrinsicFile,
                     const std::string& extrinsicFile) {
  K_ = cv::Mat::eye(3, 3, CV_64F);
  D_ = cv::Mat::zeros(4, 1, CV_64F);

  // 读取内参数据
  {  // load intrinsic
    cv::FileStorage fs(intrinsicFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      std::cerr << "Fail to open intrinsic file: " << intrinsicFile
                << std::endl;
      return;
    }
    cv::FileNode m_param = fs["mirror_parameters"];
    cv::FileNode D_param = fs["distortion_parameters"];
    cv::FileNode K_param = fs["projection_parameters"];
    if (m_param.empty() || D_param.empty() || K_param.empty()) {
      std::cerr << "Error intrinsic file: " << intrinsicFile << std::endl;
      return;
    }
    fs["image_width"] >> width_;
    fs["image_height"] >> height_;
    m_param["xi"] >> xi_;
    K_param["gamma1"] >> K_.at<double>(0, 0);
    K_param["gamma2"] >> K_.at<double>(1, 1);
    K_param["u0"] >> K_.at<double>(0, 2);
    K_param["v0"] >> K_.at<double>(1, 2);
    D_param["k1"] >> D_.at<double>(0, 0);
    D_param["k2"] >> D_.at<double>(1, 0);
    D_param["p1"] >> D_.at<double>(2, 0);
    D_param["p2"] >> D_.at<double>(3, 0);
  }

  // 读取外参数据
  {  // load extrinsic
    cv::FileStorage fs(extrinsicFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      std::cerr << "Fail to open extrinsic file: " << extrinsicFile
                << std::endl;
      return;
    }
    cv::FileNode trans_param = fs["transform"]["translation"];
    cv::FileNode rot_param = fs["transform"]["rotation"];
    if (trans_param.empty() || rot_param.empty()) {
      std::cerr << "Error extrinsic file: " << extrinsicFile << std::endl;
      return;
    }
    fs["frame_id"] >> id_;

    // T_vehicle_cam_ represents the transformation from the camera coordinate system to the vehicle coordinate system,
    // the origin of the vehicle coordinate system is the center of the vehicle.
    trans_param["x"] >> T_vehicle_cam_.translation().x();
    trans_param["y"] >> T_vehicle_cam_.translation().y();
    trans_param["z"] >> T_vehicle_cam_.translation().z();
    Eigen::Quaterniond q;
    rot_param["x"] >> q.x();
    rot_param["y"] >> q.y();
    rot_param["z"] >> q.z();
    rot_param["w"] >> q.w();
    T_vehicle_cam_.linear() = q.toRotationMatrix();
  }

  is_valid_ = true;
}


// MEICamera结构体的DebugString函数功能实现
// 打印内参、外参文件中的数据
void MEICamera::DebugString() const {
  std::cout << id_ << " : " << is_valid_ << std::endl;
  std::cout << "height: " << height_ << std::endl;
  std::cout << "width: " << width_ << std::endl;
  std::cout << "xi: " << xi_ << std::endl;
  std::cout << "K : " << K_ << std::endl;
  std::cout << "D : " << D_ << std::endl;
  std::cout << "cam_extrinsic : " << T_vehicle_cam_.matrix() << std::endl;
}

IPM::IPM() {
  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0) {
    hw_threads = 1;
  }
  hw_threads = std::max(1u, std::min(hw_threads, 8u));
  omp_set_dynamic(0);
  omp_set_num_threads(static_cast<int>(hw_threads));
}


// 添加相机（前后左右四个鱼眼相机）并加载参数
void IPM::AddCamera(const std::string& intrinsicsFile,
                    const std::string& extrinsicFile) {
  cameras_.emplace_back(intrinsicsFile, extrinsicFile);
  std::cout << "---------ipm add new camera ---------" << std::endl;
  cameras_.back().DebugString();
  lut_dirty_ = true;
}


cv::Mat IPM::GenerateIPMImage(const std::vector<cv::Mat>& images) const {
  return GenerateIPMImageImpl(images, false);
}

cv::Mat IPM::GenerateIPMImage(const std::vector<cv::Mat>& images,
                              bool preserve_labels) const {
  return GenerateIPMImageImpl(images, preserve_labels);
}

cv::Mat IPM::GenerateIPMImageImpl(const std::vector<cv::Mat>& images,
                                  bool preserve_labels) const {
  cv::Mat ipm_image;
  ipm_image.create(ipm_img_h_, ipm_img_w_, CV_8UC3);
  ipm_image.setTo(cv::Scalar(0, 0, 0));

  if (images.size() != cameras_.size() || images.empty()) {
    std::cout << "IPM not init normally!" << std::endl;
    return ipm_image;
  }

  EnsureLookupTable();
  const int total_pixels = ipm_img_h_ * ipm_img_w_;

  const bool debug_active = debug_export_weights_;
  if (debug_active) {
    PrepareDebugBuffers();
  } else {
    debug_data_ready_ = false;
  }

  std::vector<const cv::Vec3b*> cam_base(cameras_.size());
  for (size_t i = 0; i < cameras_.size(); ++i) {
    const cv::Mat& img = images[i];
    cam_base[i] = img.ptr<cv::Vec3b>(0);
  }
  const cv::Vec3b* const* cam_base_ptr = cam_base.data();

  cv::Vec3b* ipm_base = ipm_image.ptr<cv::Vec3b>(0);

  #pragma omp parallel for schedule(static, 256)
  for (int idx = 0; idx < total_pixels; ++idx) {
    const PixelLookupEntry& entry = pixel_lut_[idx];
    if (entry.count == 0) {
      continue;
    }

    const float inv = entry.inv_weight_sum;
    if (debug_active && inv > 0.0f) {
      const int y = idx / ipm_img_w_;
      const int x = idx % ipm_img_w_;
      for (uint8_t c = 0; c < entry.count; ++c) {
        const PixelContribution& contrib = entry.contributions[c];
        const size_t camera_idx = contrib.camera_idx;
        if (camera_idx >= debug_weight_maps_.size()) {
          continue;
        }
        const float normalized = contrib.weight * inv;
        debug_weight_maps_[camera_idx].at<float>(y, x) = normalized;
        if (c == 0 && camera_idx < 255u) {
          debug_dominant_map_.at<uint8_t>(y, x) =
              static_cast<uint8_t>(camera_idx);
        }
      }
    }

    if (preserve_labels) {
      const PixelContribution& contrib = entry.contributions[0];
      const size_t camera_idx = contrib.camera_idx;
      const cv::Vec3b& pixel = cam_base_ptr[camera_idx][contrib.offset];
      ipm_base[idx] = pixel;
      continue;
    }

    cv::Vec3f accum(0.0f, 0.0f, 0.0f);
    for (uint8_t c = 0; c < entry.count; ++c) {
      const PixelContribution& contrib = entry.contributions[c];
      const size_t camera_idx = contrib.camera_idx;
      const cv::Vec3b& pixel = cam_base_ptr[camera_idx][contrib.offset];
      const float w = contrib.weight;
      accum[0] += pixel[0] * w;
      accum[1] += pixel[1] * w;
      accum[2] += pixel[2] * w;
    }

    if (inv > 0.0f) {
      cv::Vec3b& dst = ipm_base[idx];
      dst[0] = cv::saturate_cast<uchar>(accum[0] * inv);
      dst[1] = cv::saturate_cast<uchar>(accum[1] * inv);
      dst[2] = cv::saturate_cast<uchar>(accum[2] * inv);
    }
  }

  if (debug_active) {
    debug_data_ready_ = true;
  }

  return ipm_image;
}

void IPM::EnsureLookupTable() const {
  if (!lut_dirty_ && !pixel_lut_.empty()) {
    return;
  }
  BuildLookupTable();
  lut_dirty_ = false;
}

void IPM::BuildLookupTable() const {
  const int total_pixels = ipm_img_h_ * ipm_img_w_;
  pixel_lut_.assign(total_pixels, PixelLookupEntry{});

  if (cameras_.empty()) {
    return;
  }

  std::vector<Eigen::Affine3d> T_cam_vehicle(cameras_.size());
  std::vector<float> center_x(cameras_.size());
  std::vector<float> center_y(cameras_.size());

  const bool apply_distance_weight =
      blending_strategy_ != BlendingStrategy::SIMPLE_AVERAGE;
  const bool apply_image_weight =
      blending_strategy_ == BlendingStrategy::WEIGHTED_ADVANCED ||
      blending_strategy_ == BlendingStrategy::HYBRID;
  const bool apply_direction_weight =
      blending_strategy_ == BlendingStrategy::WEIGHTED_SIMPLE ||
      blending_strategy_ == BlendingStrategy::WEIGHTED_ADVANCED ||
      blending_strategy_ == BlendingStrategy::HYBRID;

  for (size_t i = 0; i < cameras_.size(); ++i) {
    T_cam_vehicle[i] = cameras_[i].T_vehicle_cam_.inverse();
    center_x[i] = static_cast<float>(cameras_[i].width_) * 0.5f;
    center_y[i] = static_cast<float>(cameras_[i].height_) * 0.5f;
  }

  const float half_h = static_cast<float>(ipm_img_h_) * 0.5f;
  const float half_w = static_cast<float>(ipm_img_w_) * 0.5f;
  const double scale = pixel_scale_;

  struct Candidate {
    uint16_t camera_idx;
    uint16_t u;
    uint16_t v;
    float weight;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(cameras_.size());

  for (int u = 0; u < ipm_img_w_; ++u) {
    for (int v = 0; v < ipm_img_h_; ++v) {
      Eigen::Vector3d p_v(-(half_h - static_cast<float>(u)) * scale,
                          (half_w - static_cast<float>(v)) * scale,
                          ipm_img_height_);

      candidates.clear();

      for (size_t i = 0; i < cameras_.size(); ++i) {
        const Eigen::Vector3d p_c = T_cam_vehicle[i] * p_v;
        const double z = p_c[2];
        if (z < 0.0) {
          continue;
        }

        const double xi = cameras_[i].xi_;
        const double d = p_c.norm();
        const double denominator = d + xi * z;
        if (denominator <= 0.0) {
          continue;
        }

        const Eigen::Vector3d p_c_normalized = p_c.normalized();
        const Eigen::Vector3d p_C(p_c_normalized[0],
                                  p_c_normalized[1],
                                  p_c_normalized[2] + xi);

        if (p_C[2] <= 0.0) {
          continue;
        }

        const Eigen::Vector2d p_unitp(p_C[0] / p_C[2], p_C[1] / p_C[2]);
        const cv::Mat& K = cameras_[i].K_;
        const int uv0 = static_cast<int>(K.at<double>(0,0) * p_unitp[0] + K.at<double>(0,2));
        const int uv1 = static_cast<int>(K.at<double>(1,1) * p_unitp[1] + K.at<double>(1,2));

        if (uv0 < 0 || uv0 >= cameras_[i].width_ ||
            uv1 < 0 || uv1 >= cameras_[i].height_) {
          continue;
        }

        float dist_weight = 1.0f;
        if (apply_distance_weight) {
          dist_weight = 1.0f / (1.0f + static_cast<float>(d) * 0.1f);
        }

        float image_weight = 1.0f;
        if (apply_image_weight) {
          const float dx =
              (static_cast<float>(uv0) - center_x[i]) / center_x[i];
          const float dy =
              (static_cast<float>(uv1) - center_y[i]) / center_y[i];
          const float edge_dist = std::sqrt(dx * dx + dy * dy);
          image_weight = std::exp(-edge_dist * edge_dist * 2.0f);
        }

        const double px = p_v[0];
        const double py = p_v[1];
        float direction_weight = 1.0f;
        if (apply_direction_weight) {
          if (i == 0) {
            direction_weight =
                (px > 0.0) ? 1.0f
                           : std::exp(-static_cast<float>(std::fabs(px)) * 0.5f);
          } else if (i == 1) {
            direction_weight =
                (px < 0.0) ? 1.0f
                           : std::exp(-static_cast<float>(std::fabs(px)) * 0.5f);
          } else if (i == 2) {
            direction_weight =
                (py > 0.0) ? 1.0f
                           : std::exp(-static_cast<float>(std::fabs(py)) * 0.5f);
          } else if (i == 3) {
            direction_weight =
                (py < 0.0) ? 1.0f
                           : std::exp(-static_cast<float>(std::fabs(py)) * 0.5f);
          }
        }

        const float weight = dist_weight * image_weight * direction_weight;
        if (weight <= 1e-6f) {
          continue;
        }

        Candidate candidate;
        candidate.camera_idx = static_cast<uint16_t>(i);
        candidate.u = static_cast<uint16_t>(uv0);
        candidate.v = static_cast<uint16_t>(uv1);
        candidate.weight = weight;
        candidates.push_back(candidate);
      }

      if (candidates.empty()) {
        continue;
      }

      std::sort(candidates.begin(), candidates.end(),
                [](const Candidate& a, const Candidate& b) {
                  return a.weight > b.weight;
                });

      const int pixel_index = v * ipm_img_w_ + u;
      PixelLookupEntry entry;
      entry.count = static_cast<uint8_t>(std::min<size_t>(2, candidates.size()));

      float weight_sum = 0.0f;
      for (uint8_t idx_c = 0; idx_c < entry.count; ++idx_c) {
        const Candidate& cand = candidates[idx_c];
        PixelContribution contrib;
        contrib.camera_idx = cand.camera_idx;
        contrib.offset = static_cast<uint32_t>(cand.v) *
                         static_cast<uint32_t>(cameras_[cand.camera_idx].width_) +
                         static_cast<uint32_t>(cand.u);
        contrib.weight = cand.weight;
        entry.contributions[idx_c] = contrib;
        weight_sum += cand.weight;
      }
      if (weight_sum > 1e-6f) {
        entry.inv_weight_sum = 1.0f / weight_sum;
      }
      pixel_lut_[pixel_index] = entry;
    }
  }
}


// 预计算映射表，遍历映射表来加快IPM合成速度
// cv::Mat IPM::GenerateIPMImage(const std::vector<cv::Mat>& images) const {
//   ...
