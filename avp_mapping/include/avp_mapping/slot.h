#pragma once
#include <array>
#include <fstream>
#include <limits>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

namespace {
// parameters for slot
constexpr int kIPMImgHeight = 1000;
constexpr int kIPMImgWidth = 1000;

constexpr double kDefaultSlotWidthMeters = 4.2;
constexpr double kDefaultSlotLengthMeters = 6.4;

constexpr int kLongLineDetectPixelNumberMin = 100;
constexpr int kShortLineDetectPixelNumberMin = 150;

constexpr int kNeighborDistance = 80; // 两车位点小于这个距离视为同一个车位点
constexpr double kPixelScale = 0.02;
constexpr double kPixelScaleInv = 1.0 / kPixelScale;
constexpr double kPixelScaleSqInv = kPixelScaleInv * kPixelScaleInv;

constexpr int kShortLinePixelDistance =
    static_cast<int>(kDefaultSlotWidthMeters * kPixelScaleInv + 0.5);
constexpr int kLongLinePixelDistance =
    static_cast<int>(kDefaultSlotLengthMeters * kPixelScaleInv + 0.5);

// 本地数据集
// const uchar kDashGray = 160;
// const uchar kArrowGray = 179;
// const uchar kSlotGray = 140;
// const uchar kSlotGray1 = 139;
 
// ipm_label
const uchar kDashGray = 127;   // 虚线车道线灰度值
const uchar kArrowGray = 225;  // 箭头车道线灰度值
const uchar kSlotGray = 151;   // 车位线主灰度值
const uchar kSlotGray1 = 150;  // 车位线次灰度值（容差）

// ipm_seg
// const uchar kDashGray = 101;   // 虚线车道线灰度值
// const uchar kArrowGray = 213;  // 箭头车道线灰度值
// const uchar kSlotGray = 160;   // 车位线主灰度值
// const uchar kSlotGray1 = 159;  // 车位线次灰度值（容差）

constexpr double kToRad = M_PI / 180.0;
constexpr double kToDeg = 180.0 / M_PI;
} // end of namespace

enum SemanticLabel {
  kDashLine = 0,
  kArrowLine,
  kSlot,
  kTotalLabelNum // flag that indicate label num
};

struct Slot {
  Eigen::Vector3d corners_[4]; // 由四个3d坐标的点构成
  bool occupied = false;       // 车位占用状态
  float occupancy_score = 0.f; // 占用置信度
};

// slot cornerPoints
class CornerPoint {
public:
  CornerPoint() : CornerPoint({0,0,0}) {}
  explicit CornerPoint(const Eigen::Vector3d &pt) : center_(pt), count_(1) {}
  // return true if the point is absorbed (distance < dist)
  bool absorb(const Eigen::Vector3d &pt, const double dist) {
    if ((pt - center()).norm() < dist) {
      center_ += pt;
      ++count_;
      return true;
    }
    return false;
  }
  Eigen::Vector3d center() const {
    return center_ / count_;
  }
private:
  Eigen::Vector3d center_;
  unsigned int count_;
};
// image processing utils for slot detection
cv::Mat skeletonize(const cv::Mat &img);
void removeIsolatedPixels(cv::Mat &src, int min_neighbors = 1);
bool isSlotLongLine(const cv::Mat &line_img, const cv::Point2f &start,
                    const cv::Point2f &dir,
                    const cv::Mat &vehicle_mask = cv::Mat{});
bool isSlotShortLine(const cv::Point2f &point1, const cv::Point2f &point2,
                     const cv::Mat &mask,
                     const cv::Mat &vehicle_mask = cv::Mat{});

struct SlotModelParams {
  float width_mean_m = static_cast<float>(kDefaultSlotWidthMeters);
  float length_mean_m = static_cast<float>(kDefaultSlotLengthMeters);
  float width_sigma_m = 0.4f;
  float length_sigma_m = 0.6f;
  float lambda_prior = 0.8f;
  float lambda_visibility = 0.25f;
  float huber_delta_px = 2.5f;
  float energy_accept_threshold = 80.0f;
  int samples_per_edge = 56;
  float heading_weight = 1.0f;
};

struct SlotHypothesis {
  cv::Point2f center_px{0.f, 0.f};
  float theta_rad = 0.f;
  float width_m = static_cast<float>(kDefaultSlotWidthMeters);
  float length_m = static_cast<float>(kDefaultSlotLengthMeters);
  std::array<float, 3> visibility_logits{{0.f, 0.f, 0.f}};
  std::array<float, 3> visibility{{0.5f, 0.5f, 0.5f}};
  float heading_prior_rad = 0.f;
  bool has_heading_prior = false;
  float energy = std::numeric_limits<float>::max();
  std::array<cv::Point2f, 4> corners{};
  bool valid = false;
};

struct SlotDetectionStats {
  size_t frame_id = 0;
  size_t num_hypotheses = 0;
  size_t num_after_nms = 0;
};

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>,
           std::vector<float>, std::vector<std::array<float, 3>>,
           SlotDetectionStats>
detectSlot(const cv::Mat &slot_img, const cv::Mat &vehicle_mask,
           const std::vector<cv::Vec4i> &lines);

class ParametricSlotFitter {
public:
  ParametricSlotFitter(const cv::Mat &distance_field, float pixel_scale);

  SlotHypothesis initializeFromCorners(const std::array<cv::Point2f, 4> &corners) const;
  SlotHypothesis optimize(const SlotHypothesis &initial,
                          const SlotModelParams &params) const;

  static float computeIoU(const SlotHypothesis &a, const SlotHypothesis &b, float pixel_scale);

private:
  SlotHypothesis clampHypothesis(const SlotHypothesis &hyp) const;
  void updateCorners(SlotHypothesis *hyp) const;
  void updateVisibility(SlotHypothesis *hyp) const;
  float computeEnergy(const SlotHypothesis &hyp, const SlotModelParams &params) const;
  float sampleEdgeEnergy(const cv::Point2f &start, const cv::Point2f &end,
                         float visibility, const SlotModelParams &params) const;
  float huberLoss(float value, float delta) const;
  float bilinearDistance(const cv::Point2f &pt) const;

  cv::Mat distance_field_;
  const float pixel_scale_;
};
// IO method for data serialization and deserialization
namespace io{
template <typename T>
void loadVector(std::ifstream &ifs, std::vector<T> &vec) {
  size_t size;
  ifs.read(reinterpret_cast<char *>(&size), sizeof(size));
  vec.resize(size);
  ifs.read(reinterpret_cast<char *>(vec.data()), size * sizeof(T));
}
template <typename T>
void saveVector(std::ofstream &ofs, const std::vector<T> &vec) {
  size_t size = vec.size();
  ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
  ofs.write(reinterpret_cast<const char *>(vec.data()), size * sizeof(T));
}
}
