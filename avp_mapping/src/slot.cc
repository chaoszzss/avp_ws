#include "../include/avp_mapping/slot.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>

namespace {
constexpr const char kSlotOutputDir[] =
    "/home/chaoszzss/avp_ws/src/avp_mapping/src/AVPMapping/output/"
    "Quantitative_indicators/slot_det1";

/// 计算两条车道线中点之间的距离
double computeCenterDistance(cv::Vec4i line1, cv::Vec4i line2) {
  cv::Point2f p1 ((line1[0] + line1[2]) / 2.0f,
                  (line1[1] + line1[3]) / 2.0f);
  cv::Point2f p2 ((line2[0] + line2[2]) / 2.0f,
                  (line2[1] + line2[3]) / 2.0f);
  return cv::norm(p1 - p2);
}

// 计算两条线段四个端点组合的最小距离
double computeSegmentDistance(cv::Vec4i line1, cv::Vec4i line2) {
  cv::Point2f p1(line1[0], line1[1]), p2(line1[2], line1[3]);
  cv::Point2f p3(line2[0], line2[1]), p4(line2[2], line2[3]);
  
  double min_dist = DBL_MAX;
  // 计算四条端点组合的最小距离
  min_dist = std::min(min_dist, cv::norm(p1 - p3));
  min_dist = std::min(min_dist, cv::norm(p1 - p4));
  min_dist = std::min(min_dist, cv::norm(p2 - p3));
  min_dist = std::min(min_dist, cv::norm(p2 - p4));
  return min_dist;
}

// 计算点p3相对于线段p1-p2的方向
float direction(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3) {
    return (p3.x - p1.x) * (p2.y - p1.y) - (p2.x - p1.x) * (p3.y - p1.y);
}

// 修改线段相交判断函数
bool segmentsIntersect(const cv::Vec4i& line1, const cv::Vec4i& line2) {
    cv::Point2f a0(line1[0], line1[1]), a1(line1[2], line1[3]);
    cv::Point2f b0(line2[0], line2[1]), b1(line2[2], line2[3]);

    // 优化包围盒测试
    auto bboxOverlap = [](float min1, float max1, float min2, float max2) {
        return max1 >= min2 && max2 >= min1;
    };
    
    bool overlap_x = bboxOverlap(
        std::min(a0.x, a1.x), std::max(a0.x, a1.x),
        std::min(b0.x, b1.x), std::max(b0.x, b1.x));
    
    bool overlap_y = bboxOverlap(
        std::min(a0.y, a1.y), std::max(a0.y, a1.y),
        std::min(b0.y, b1.y), std::max(b0.y, b1.y));
    
    if (!overlap_x || !overlap_y) return false;

    // 改进的方向计算（带容差）
    const float eps = 1e-5f;
    auto cross = [](const cv::Point2f& p1, const cv::Point2f& p2) {
        return p1.x*p2.y - p1.y*p2.x;
    };

    cv::Point2f r = a1 - a0;
    cv::Point2f s = b1 - b0;
    cv::Point2f v = b0 - a0;

    float crs = cross(r, s);
    float t = cross(v, s) / crs;
    float u = cross(v, r) / crs;

    // 允许端点接触（t和u在[-eps, 1+eps]范围）
    return (t > -eps) && (t < 1.0f + eps) && 
           (u > -eps) && (u < 1.0f + eps);
}

/// 计算两条车道线的交点（返回交点坐标）
cv::Point2f computeIntersect(cv::Vec4i line1, cv::Vec4i line2) {
  int x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
  int x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];
  float denom = (float)((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
  cv::Point2f intersect( (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4),
                         (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4));
  return intersect / denom;
  }

std::array<cv::Point2f, 4> orderCorners(const std::array<cv::Point2f, 4> &corners) {
  cv::Point2f centroid(0.f, 0.f);
  for (const auto &pt : corners) {
    centroid += pt;
  }
  centroid *= 0.25f;

  std::array<int, 4> indices{{0, 1, 2, 3}};
  std::sort(indices.begin(), indices.end(),
            [&](int lhs, int rhs) {
              float angle_l = std::atan2(corners[lhs].y - centroid.y,
                                         corners[lhs].x - centroid.x);
              float angle_r = std::atan2(corners[rhs].y - centroid.y,
                                         corners[rhs].x - centroid.x);
              return angle_l < angle_r;
            });

  std::array<cv::Point2f, 4> ordered{};
  for (size_t i = 0; i < indices.size(); ++i) {
    ordered[i] = corners[indices[i]];
  }

  // Rotate so the first point is the top-left (min x+y) to enforce consistency.
  int best_idx = 0;
  float best_score = std::numeric_limits<float>::max();
  for (int i = 0; i < 4; ++i) {
    float score = ordered[i].x + ordered[i].y;
    if (score < best_score) {
      best_score = score;
      best_idx = i;
    }
  }
  std::array<cv::Point2f, 4> rotated{};
  for (int i = 0; i < 4; ++i) {
    rotated[i] = ordered[(best_idx + i) % 4];
  }
  return rotated;
}

std::vector<SlotHypothesis> nonMaxSuppression(const std::vector<SlotHypothesis> &slots,
                                              float iou_threshold,
                                              float pixel_scale) {
  std::vector<SlotHypothesis> sorted = slots;
  std::sort(sorted.begin(), sorted.end(),
            [](const SlotHypothesis &lhs, const SlotHypothesis &rhs) {
              return lhs.energy < rhs.energy;
            });
  std::vector<SlotHypothesis> kept;
  for (const auto &slot : sorted) {
    bool suppressed = false;
    for (const auto &existing : kept) {
      const float overlap = ParametricSlotFitter::computeIoU(slot, existing, pixel_scale);
      if (overlap > iou_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(slot);
    }
  }
  return kept;
}

inline float sigmoidf(float x) {
  return 1.f / (1.f + std::exp(-x));
}

inline float clampf(float value, float min_v, float max_v) {
  return std::max(min_v, std::min(max_v, value));
}

inline float wrapAngle(float angle) {
  while (angle <= -static_cast<float>(M_PI)) {
    angle += 2.f * static_cast<float>(M_PI);
  }
  while (angle > static_cast<float>(M_PI)) {
    angle -= 2.f * static_cast<float>(M_PI);
  }
  return angle;
}

inline float angleDistance(float lhs, float rhs) {
  const float diff = wrapAngle(lhs - rhs);
  return std::fabs(diff);
}

bool slotDebugEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("AVP_SLOT_DEBUG");
    if (!env) {
      return true;  // 默认启用 CSV 采集
    }
    return std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

void appendSlotDebugLog(size_t frame_id, const std::vector<SlotHypothesis> &slots) {
  if (!slotDebugEnabled() || slots.empty()) {
    return;
  }
  static std::mutex log_mutex;
  std::lock_guard<std::mutex> lock(log_mutex);
  static std::ofstream ofs;
  if (!ofs.is_open()) {
    std::filesystem::create_directories(kSlotOutputDir);
    const auto log_path = std::filesystem::path(kSlotOutputDir) / "slot_debug.csv";
    ofs.open(log_path, std::ios::app);
    if (!ofs) {
      return;
    }
    if (ofs.tellp() == 0) {
      ofs << "frame,index,cx_px,cy_px,width_m,length_m,theta_deg,energy,vis_side1,"
             "vis_side2,vis_stop\n";
    }
  }
  ofs << std::fixed << std::setprecision(3);
  for (size_t i = 0; i < slots.size(); ++i) {
    const auto &slot = slots[i];
    ofs << frame_id << ',' << i << ',' << slot.center_px.x << ',' << slot.center_px.y
        << ',' << slot.width_m << ',' << slot.length_m << ','
        << slot.theta_rad * kToDeg << ',' << slot.energy << ','
        << slot.visibility[0] << ',' << slot.visibility[1] << ','
        << slot.visibility[2] << '\n';
  }
  ofs.flush();
}

size_t nextSlotFrameId() {
  static size_t counter = 0;
  return counter++;
}

} // end of namespace

// 保留边角、细小区域 和骨架
cv::Mat skeletonize(const cv::Mat &img) {
  cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
  cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
  cv::Mat temp, eroded;
  do {
    cv::erode(img, eroded, element);
    cv::dilate(eroded, temp, element);
    cv::subtract(img, temp, temp);  // 细小区域
    cv::bitwise_or(skel, temp, skel);
    eroded.copyTo(img);
  } while (0 != cv::countNonZero(img));
  return skel;
}

/// 去除孤立像素
void removeIsolatedPixels(cv::Mat &src, int min_neighbors) {
  CV_Assert(src.type() == CV_8UC1);  // 确保单通道灰度图
  cv::Mat dst = src.clone();
  for (int y = 1; y < src.rows - 1; ++y) {
    for (int x = 1; x < src.cols - 1; ++x) {
      if (src.at<uchar>(y, x) > 0) {  // 如果是前景图像
        int neighbor_count = 0;       // 相邻前景像素的计数器
        // 检查8邻域内的像素
        for (int ny = -1; ny <= 1; ++ny) {
          for (int nx = -1; nx <= 1; ++nx) {
            if (ny == 0 && nx == 0) continue;  // 跳过自己
            if (src.at<uchar>(y + ny, x + nx) > 0) {
              ++neighbor_count;  // 增加邻居数
            }
          }
        }
        // 如果相邻前景像素小于阈值，则认为该点是孤立的，并将其去除
        if (neighbor_count < min_neighbors) {
          dst.at<uchar>(y, x) = 0;
        }
      }
    }
  }
  src = dst;
}

ParametricSlotFitter::ParametricSlotFitter(const cv::Mat &distance_field,
                                           float pixel_scale)
    : pixel_scale_(pixel_scale) {
  CV_Assert(distance_field.type() == CV_32FC1);
  distance_field_ = distance_field.clone();
}

SlotHypothesis ParametricSlotFitter::initializeFromCorners(
    const std::array<cv::Point2f, 4> &corners) const {
  SlotHypothesis hyp;
  cv::Point2f center(0.f, 0.f);
  for (const auto &corner : corners) {
    center += corner;
  }
  hyp.center_px = center * 0.25f;

  auto edgeLength = [](const cv::Point2f &a, const cv::Point2f &b) {
    return cv::norm(a - b);
  };
  const float len01 = edgeLength(corners[0], corners[1]);
  const float len12 = edgeLength(corners[1], corners[2]);
  const float len23 = edgeLength(corners[2], corners[3]);
  const float len30 = edgeLength(corners[3], corners[0]);

  std::array<float, 4> lengths{len01, len12, len23, len30};
  const auto max_it =
      std::max_element(lengths.begin(), lengths.end());
  const auto min_it =
      std::min_element(lengths.begin(), lengths.end());

  hyp.length_m = *max_it * pixel_scale_;
  hyp.width_m = *min_it * pixel_scale_;

  const int max_idx = static_cast<int>(std::distance(lengths.begin(), max_it));
  cv::Point2f direction;
  switch (max_idx) {
    case 0: direction = corners[1] - corners[0]; break;
    case 1: direction = corners[2] - corners[1]; break;
    case 2: direction = corners[3] - corners[2]; break;
    default: direction = corners[0] - corners[3]; break;
  }
  hyp.theta_rad = std::atan2(direction.y, direction.x);
  updateCorners(&hyp);
  hyp.energy = computeEnergy(hyp, SlotModelParams{});
  hyp.valid = true;
  return hyp;
}

SlotHypothesis ParametricSlotFitter::optimize(
    const SlotHypothesis &initial,
    const SlotModelParams &params) const {
  SlotHypothesis current = clampHypothesis(initial);
  updateCorners(&current);
  current.energy = computeEnergy(current, params);
  SlotHypothesis best = current;

  const int max_iterations = 100;  // 最大迭代次数
  std::array<float, 8> deltas{{1.0f, 1.0f, 0.01f, 0.08f, 0.1f, 0.5f, 0.5f, 0.5f}};
  std::array<float, 8> learning_rates{{0.05f, 0.05f, 0.4f, 0.1f, 0.1f, 0.4f, 0.4f, 0.4f}};

  for (int iter = 0; iter < max_iterations; ++iter) {
    std::array<float, 8> gradient{};

    for (int idx = 0; idx < gradient.size(); ++idx) {
      SlotHypothesis plus = current;
      SlotHypothesis minus = current;
      float delta = deltas[idx];

      auto adjustParam = [idx, delta](SlotHypothesis *hyp, float sign) {
        switch (idx) {
          case 0: hyp->center_px.x += sign * delta; break;
          case 1: hyp->center_px.y += sign * delta; break;
          case 2: hyp->theta_rad += sign * delta; break;
          case 3: hyp->width_m += sign * delta; break;
          case 4: hyp->length_m += sign * delta; break;
          case 5: hyp->visibility_logits[0] += sign * delta; break;
          case 6: hyp->visibility_logits[1] += sign * delta; break;
          case 7: hyp->visibility_logits[2] += sign * delta; break;
          default: break;
        }
      };

      adjustParam(&plus, 1.f);
      adjustParam(&minus, -1.f);
      plus = clampHypothesis(plus);
      minus = clampHypothesis(minus);
      updateCorners(&plus);
      updateCorners(&minus);
      plus.energy = computeEnergy(plus, params);
      minus.energy = computeEnergy(minus, params);

      gradient[idx] = (plus.energy - minus.energy) / (2.f * delta);
    }

    auto applyUpdate = [&](SlotHypothesis *hyp, const std::array<float, 8> &grad,
                           const std::array<float, 8> &lr) {
      hyp->center_px.x -= lr[0] * grad[0];
      hyp->center_px.y -= lr[1] * grad[1];
      hyp->theta_rad -= lr[2] * grad[2];
      hyp->width_m -= lr[3] * grad[3];
      hyp->length_m -= lr[4] * grad[4];
      hyp->visibility_logits[0] -= lr[5] * grad[5];
      hyp->visibility_logits[1] -= lr[6] * grad[6];
      hyp->visibility_logits[2] -= lr[7] * grad[7];
    };

    SlotHypothesis proposal = current;
    applyUpdate(&proposal, gradient, learning_rates);
    proposal = clampHypothesis(proposal);
    updateCorners(&proposal);
    proposal.energy = computeEnergy(proposal, params);

    if (proposal.energy < current.energy) {
      current = proposal;
      if (proposal.energy < best.energy) {
        best = proposal;
      }
    } else {
      // 若未改进，适度降低学习率避免震荡
      for (auto &lr : learning_rates) {
        lr *= 0.8f;
      }
      if (*std::max_element(learning_rates.begin(), learning_rates.end()) < 1e-4f) {
        break;
      }
    }
  }

  best.valid = best.energy < params.energy_accept_threshold;
  return best;
}

float ParametricSlotFitter::computeIoU(const SlotHypothesis &a,
                                       const SlotHypothesis &b,
                                       float pixel_scale) {
  const float width_a = a.width_m / pixel_scale;
  const float length_a = a.length_m / pixel_scale;
  const float width_b = b.width_m / pixel_scale;
  const float length_b = b.length_m / pixel_scale;
  const float angle_a = a.theta_rad * 180.f / static_cast<float>(M_PI);
  const float angle_b = b.theta_rad * 180.f / static_cast<float>(M_PI);

  cv::RotatedRect rect_a(a.center_px, cv::Size2f(length_a, width_a), angle_a);
  cv::RotatedRect rect_b(b.center_px, cv::Size2f(length_b, width_b), angle_b);

  std::vector<cv::Point2f> intersection;
  auto result = cv::rotatedRectangleIntersection(rect_a, rect_b, intersection);

  float area_a = length_a * width_a;
  float area_b = length_b * width_b;
  float inter_area = 0.f;

  if (result == cv::INTERSECT_FULL) {
    inter_area = std::min(area_a, area_b);
  } else if (result == cv::INTERSECT_PARTIAL && !intersection.empty()) {
    inter_area = std::fabs(cv::contourArea(intersection));
  }

  const float union_area = area_a + area_b - inter_area + 1e-6f;
  return inter_area / union_area;
}

SlotHypothesis ParametricSlotFitter::clampHypothesis(const SlotHypothesis &hyp) const {
  SlotHypothesis clamped = hyp;
  const float max_x = static_cast<float>(distance_field_.cols - 2);
  const float max_y = static_cast<float>(distance_field_.rows - 2);
  clamped.center_px.x = clampf(clamped.center_px.x, 1.f, max_x);
  clamped.center_px.y = clampf(clamped.center_px.y, 1.f, max_y);
  // weight：4.2m  height：6.4m
  // const float min_width = static_cast<float>(kDefaultSlotWidthMeters) - 0.8f;   // 3.4m
  // const float max_width = static_cast<float>(kDefaultSlotWidthMeters) + 0.8f;   // 5.0m
  // const float min_length = static_cast<float>(kDefaultSlotLengthMeters) - 0.8f; // 5.6m
  // const float max_length = static_cast<float>(kDefaultSlotLengthMeters) + 0.8f; // 7.2m
  const float min_width = static_cast<float>(kDefaultSlotWidthMeters) - 0.2f;   // 4m
  const float max_width = static_cast<float>(kDefaultSlotWidthMeters) + 0.2f;   // 4.4m
  const float min_length = static_cast<float>(kDefaultSlotLengthMeters) - 0.2f; // 6.2m
  const float max_length = static_cast<float>(kDefaultSlotLengthMeters) + 0.2f; // 6.6m
  clamped.width_m = clampf(clamped.width_m, min_width, max_width);
  clamped.length_m = clampf(clamped.length_m, min_length, max_length);
  clamped.theta_rad = wrapAngle(clamped.theta_rad);
  return clamped;
}

void ParametricSlotFitter::updateCorners(SlotHypothesis *hyp) const {
  const float half_long = 0.5f * hyp->length_m / pixel_scale_;
  const float half_short = 0.5f * hyp->width_m / pixel_scale_;

  const cv::Point2f u(std::cos(hyp->theta_rad), std::sin(hyp->theta_rad));
  const cv::Point2f v(-u.y, u.x);

  hyp->corners[0] = hyp->center_px - v * half_short - u * half_long;
  hyp->corners[1] = hyp->center_px - v * half_short + u * half_long;
  hyp->corners[2] = hyp->center_px + v * half_short + u * half_long;
  hyp->corners[3] = hyp->center_px + v * half_short - u * half_long;
  hyp->corners = orderCorners(hyp->corners);
  updateVisibility(hyp);
}

void ParametricSlotFitter::updateVisibility(SlotHypothesis *hyp) const {
  for (int i = 0; i < 3; ++i) {
    hyp->visibility[i] = sigmoidf(hyp->visibility_logits[i]);
  }
}

float ParametricSlotFitter::computeEnergy(const SlotHypothesis &hyp,
                                          const SlotModelParams &params) const {
  const float side_energy_1 =
      sampleEdgeEnergy(hyp.corners[0], hyp.corners[1], hyp.visibility[0], params);
  const float side_energy_2 =
      sampleEdgeEnergy(hyp.corners[2], hyp.corners[3], hyp.visibility[1], params);
  const float stop_energy =
      sampleEdgeEnergy(hyp.corners[1], hyp.corners[2], hyp.visibility[2], params);

  const float data_term = side_energy_1 + side_energy_2 + stop_energy;
  const float width_prior =
      std::pow((hyp.width_m - params.width_mean_m) / (params.width_sigma_m + 1e-6f), 2.f);
  const float length_prior =
      std::pow((hyp.length_m - params.length_mean_m) / (params.length_sigma_m + 1e-6f), 2.f);
  float heading_prior = 0.f;
  if (hyp.has_heading_prior) {
    const float angle_error = angleDistance(hyp.theta_rad, hyp.heading_prior_rad);
    heading_prior = params.heading_weight * angle_error * angle_error;
  }

  return data_term + params.lambda_prior * (width_prior + length_prior) + heading_prior;
}

float ParametricSlotFitter::sampleEdgeEnergy(const cv::Point2f &start,
                                             const cv::Point2f &end,
                                             float visibility,
                                             const SlotModelParams &params) const {
  const int samples = std::max(4, params.samples_per_edge);
  const cv::Point2f direction = (end - start) / static_cast<float>(samples - 1);
  float energy = 0.f;
  for (int i = 0; i < samples; ++i) {
    const cv::Point2f sample = start + direction * static_cast<float>(i);
    const float dist = bilinearDistance(sample);
    energy += visibility * huberLoss(dist, params.huber_delta_px) +
              params.lambda_visibility * (1.f - visibility);
  }
  return energy;
}

float ParametricSlotFitter::huberLoss(float value, float delta) const {
  const float abs_v = std::fabs(value);
  if (abs_v <= delta) {
    return 0.5f * value * value;
  }
  return delta * (abs_v - 0.5f * delta);
}

float ParametricSlotFitter::bilinearDistance(const cv::Point2f &pt) const {
  const float x = clampf(pt.x, 0.f, static_cast<float>(distance_field_.cols - 1));
  const float y = clampf(pt.y, 0.f, static_cast<float>(distance_field_.rows - 1));
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, distance_field_.cols - 1);
  const int y1 = std::min(y0 + 1, distance_field_.rows - 1);
  const float dx = x - static_cast<float>(x0);
  const float dy = y - static_cast<float>(y0);

  const float top = distance_field_.at<float>(y0, x0) * (1.f - dx) +
                    distance_field_.at<float>(y0, x1) * dx;
  const float bottom = distance_field_.at<float>(y1, x0) * (1.f - dx) +
                       distance_field_.at<float>(y1, x1) * dx;
  return top * (1.f - dy) + bottom * dy;
}


/// 判断是否为短车道线
bool isSlotShortLine(const cv::Point2f &point1,
                     const cv::Point2f &point2,
                     const cv::Mat &image,
                     const cv::Mat &vehicle_mask) {
  cv::LineIterator it(image, point1, point2, 4);
  int positiveIndex = 0;
  int visibleSamples = 0;
  int occludedSamples = 0;
  const bool use_vehicle_mask = !vehicle_mask.empty();
  const double len = cv::norm(point1 - point2);
  const double expected = static_cast<double>(kShortLinePixelDistance);
  const double delta = std::max(8.0, 0.15 * expected);
  if (std::fabs(len - kShortLinePixelDistance) < delta && it.count > 0) {
    for (int i = 0; i < it.count; ++i, ++it) {
      const cv::Point pt(cvRound(it.pos().x), cvRound(it.pos().y));
      if (pt.x < 0 || pt.x >= image.cols || pt.y < 0 || pt.y >= image.rows) {
        continue;
      }
      const bool occluded =
          use_vehicle_mask && vehicle_mask.at<uchar>(pt) > 0;
      if (occluded) {
        ++occludedSamples;
      } else {
        ++visibleSamples;
        if (image.at<uchar>(pt) > 0) {
          ++positiveIndex;
        }
      }
    }
    if (visibleSamples == 0) {
      return occludedSamples > 0;
    }
    const double occlusion_ratio =
        occludedSamples / std::max(1.0, static_cast<double>(visibleSamples + occludedSamples));
    const int base_required =
        std::max(kShortLineDetectPixelNumberMin / 2,
                 static_cast<int>(0.55 * static_cast<double>(visibleSamples)));
    const int adjusted_required =
        static_cast<int>(std::ceil(base_required * (1.0 - 0.5 * occlusion_ratio)));
    if (positiveIndex >= adjusted_required) {
      return true;
    }
  }
  return false;
}

/// 判断是否为长车道线
bool isSlotLongLine(const cv::Mat &line_img,
                    const cv::Mat &vehicle_mask,
                    const cv::Point2f &start,
                    const cv::Point2f &dir) {
  int cnt{0};
  int visible{0};
  int occluded{0};
  const bool use_vehicle_mask = !vehicle_mask.empty();
  cv::Point2f pt(start);
  for (int l = 1; l < kLongLinePixelDistance; ++l) {
    pt += dir;
    if (pt.y <= 0 || pt.y >= kIPMImgHeight ||
        pt.x <= 0 || pt.x >= kIPMImgWidth) {
      continue;
    }
    const cv::Point current(cvRound(pt.x), cvRound(pt.y));
    const bool occluded_here =
        use_vehicle_mask && vehicle_mask.at<uchar>(current) > 0;
    if (occluded_here) {
      ++occluded;
    } else {
      ++visible;
      if (line_img.at<uchar>(current) == 254) {
        ++cnt;
      }
    }
  }
  if (visible == 0) {
    return occluded > 0;
  }
  const double occlusion_ratio =
      occluded / std::max(1.0, static_cast<double>(visible + occluded));
  const int base_required =
      std::max(kLongLineDetectPixelNumberMin / 2,
               static_cast<int>(0.55 * static_cast<double>(visible)));
  const int adjusted_required =
      static_cast<int>(std::ceil(base_required * (1.0 - 0.5 * occlusion_ratio)));
  return cnt >= adjusted_required;
}

/// 检测车位
std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>,
           std::vector<float>, std::vector<std::array<float, 3>>,
           SlotDetectionStats>
detectSlot(const cv::Mat &slot_img, const cv::Mat &vehicle_mask,
           const std::vector<cv::Vec4i> &lines) {
  const size_t frame_id = nextSlotFrameId();
  std::vector<cv::Point2f> corners;
  for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
    for (int j = i + 1; j < static_cast<int>(lines.size()); ++j) {
      const double distance = computeCenterDistance(lines[i], lines[j]);
      if (distance > 320) {
        continue;
      }
      const int x1 = lines[i][0], y1 = lines[i][1], x2 = lines[i][2], y2 = lines[i][3];
      const int x3 = lines[j][0], y3 = lines[j][1], x4 = lines[j][2], y4 = lines[j][3];
      const double v1x = x2 - x1;
      const double v1y = y2 - y1;
      const double v2x = x4 - x3;
      const double v2y = y4 - y3;
      const double dotProduct = v1x * v2x + v1y * v2y;
      const double v1_norm = std::sqrt(v1x * v1x + v1y * v1y);
      const double v2_norm = std::sqrt(v2x * v2x + v2y * v2y);
      const double cos_theta = dotProduct / (v1_norm * v2_norm + 1e-6);
      if (std::fabs(cos_theta) <= 0.005f) {
        corners.push_back(computeIntersect(lines[i], lines[j]));
      }
    }
  }

  const float min_distance = kNeighborDistance;
  std::vector<cv::Point2f> filtered_corners;
  filtered_corners.reserve(corners.size());
  for (const auto &corner : corners) {
    bool duplicate = false;
    for (const auto &existing : filtered_corners) {
      if (cv::norm(corner - existing) < min_distance) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      filtered_corners.push_back(corner);
    }
  }
  corners = std::move(filtered_corners);

  if (slot_img.empty() || corners.empty()) {
    SlotDetectionStats detection_stats;
    detection_stats.frame_id = frame_id;
    detection_stats.num_hypotheses = 0;
    detection_stats.num_after_nms = 0;
    return std::make_tuple(corners, std::vector<cv::Point2f>{}, std::vector<float>{},
                           std::vector<std::array<float, 3>>{}, detection_stats);
  }

  cv::Mat binary_mask;
  cv::threshold(slot_img, binary_mask, 1, 255, cv::THRESH_BINARY);
  cv::Mat dt_source(binary_mask.size(), CV_8UC1, cv::Scalar(255));
  dt_source.setTo(0, binary_mask > 0);
  cv::Mat distance_field;
  cv::distanceTransform(dt_source, distance_field, cv::DIST_L2, 5);

  ParametricSlotFitter fitter(distance_field, static_cast<float>(kPixelScale));
  SlotModelParams params;
  params.width_sigma_m = 0.2f;            // 减小标准差，提高宽度精度要求
  params.length_sigma_m = 0.4f;           // 减小标准差，提高长度精度要求
  params.lambda_prior = 1.2f;             // 增加先验权重，更严格约束尺寸
  params.lambda_visibility = 0.25f;
  params.huber_delta_px = 2.5f;
  params.energy_accept_threshold = 60.f;   //降低能量阈值，只接受高质量拟合
  params.samples_per_edge = 56;

  auto estimateDominantOrientation = [](const std::vector<cv::Vec4i> &input_lines) {
    if (input_lines.empty()) {
      return 0.f;
    }
    constexpr int kBins = 18;
    std::array<int, kBins> histogram{};
    for (const auto &line : input_lines) {
      const float angle = std::atan2(static_cast<float>(line[3] - line[1]),
                                     static_cast<float>(line[2] - line[0]));
      float normalized = std::fmod(angle, static_cast<float>(M_PI));
      if (normalized < 0.f) {
        normalized += static_cast<float>(M_PI);
      }
      int bin = static_cast<int>(normalized / static_cast<float>(M_PI) * kBins);
      bin = std::max(0, std::min(kBins - 1, bin));
      histogram[bin]++;
    }
    const auto best = std::max_element(histogram.begin(), histogram.end());
    const int idx = static_cast<int>(std::distance(histogram.begin(), best));
    const float bin_size = static_cast<float>(M_PI) / kBins;
    return idx * bin_size;
  };

  const float dominant_orientation = estimateDominantOrientation(lines);
  std::vector<SlotHypothesis> hypotheses;

  for (int i = 0; i < static_cast<int>(corners.size()); ++i) {
    for (int j = i + 1; j < static_cast<int>(corners.size()); ++j) {
      if (!isSlotShortLine(corners[i], corners[j], slot_img, vehicle_mask)) {
        continue;
      }
      const cv::Point2f width_vec = corners[j] - corners[i];
      const double width_length = cv::norm(width_vec);
      if (width_length < 1e-3) {
        continue;
      }
      const cv::Point2f width_dir = width_vec / static_cast<float>(width_length);
      const cv::Point2f length_dir(-width_dir.y, width_dir.x);
      const cv::Point2f midpoint = 0.5f * (corners[i] + corners[j]);
      const float width_m = clampf(
          static_cast<float>(width_length) * static_cast<float>(kPixelScale),
          static_cast<float>(kDefaultSlotWidthMeters) - 0.6f,
          static_cast<float>(kDefaultSlotWidthMeters) + 0.6f);
      const float half_length_px =
          0.5f * params.length_mean_m / static_cast<float>(kPixelScale);

      std::array<cv::Point2f, 2> offset_candidates{length_dir, -length_dir};
      for (const auto &dir_candidate : offset_candidates) {
        SlotHypothesis candidate;
        const cv::Point2f axis = dir_candidate;
        candidate.center_px = midpoint + axis * half_length_px;
        candidate.theta_rad = std::atan2(axis.y, axis.x);
        candidate.width_m = width_m;
        candidate.length_m = params.length_mean_m;
        candidate.heading_prior_rad = candidate.theta_rad;
        candidate.has_heading_prior = true;
        candidate.visibility_logits = {0.8f, 0.8f, 0.2f};
        candidate.visibility = {sigmoidf(candidate.visibility_logits[0]),
                                sigmoidf(candidate.visibility_logits[1]),
                                sigmoidf(candidate.visibility_logits[2])};

        const bool long_support_first = isSlotLongLine(slot_img, vehicle_mask, corners[i], dir_candidate);
        const bool long_support_second = isSlotLongLine(slot_img, vehicle_mask, corners[j], dir_candidate);
        if (!long_support_first || !long_support_second) {
          continue;
        }

        SlotHypothesis refined = fitter.optimize(candidate, params);
        if (!refined.valid) {
          continue;
        }

        if (!std::isfinite(refined.energy)) {
          continue;
        }

        const float deviation = angleDistance(refined.theta_rad, refined.heading_prior_rad);
        const float max_deviation =
            lines.empty() ? static_cast<float>(30.0 * kToRad)
                          : static_cast<float>(20.0 * kToRad);
        if (deviation > max_deviation) {
          continue;
        }
        hypotheses.push_back(refined);
      }
    }
  }

  const float iou_threshold = 0.65f;
  std::vector<SlotHypothesis> filtered =
      nonMaxSuppression(hypotheses, iou_threshold, static_cast<float>(kPixelScale));

  std::vector<cv::Point2f> slot_points;
  slot_points.reserve(filtered.size() * 4);
  std::vector<float> slot_energies;
  slot_energies.reserve(filtered.size());
  std::vector<std::array<float, 3>> slot_visibilities;
  slot_visibilities.reserve(filtered.size());
  for (auto &slot : filtered) {
    for (const auto &corner : slot.corners) {
      slot_points.push_back(corner);
    }
    slot_energies.push_back(slot.energy);
    slot_visibilities.push_back(slot.visibility);
  }

  appendSlotDebugLog(frame_id, filtered);
  SlotDetectionStats detection_stats;
  detection_stats.frame_id = frame_id;
  detection_stats.num_hypotheses = hypotheses.size();
  detection_stats.num_after_nms = filtered.size();
  return std::make_tuple(corners, slot_points, slot_energies, slot_visibilities,
                         detection_stats);
}
