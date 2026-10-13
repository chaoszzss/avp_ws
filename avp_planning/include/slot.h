#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace {
// parameters for slot
constexpr int kIPMImgHeight = 1000;
constexpr int kIPMImgWidth = 1000;

constexpr int kShortLinePixelDistance = 210;
constexpr int kLongLinePixelDistance = 320;
constexpr int kLongLineDetectPixelNumberMin = 200;
constexpr int kShortLineDetectPixelNumberMin = 50;

constexpr int kNeighborDistance = 100;  // 两车位点小于这个距离视为同一个车位点
constexpr double kPixelScale = 0.02;
constexpr double kPixelScaleInv = 1.0 / kPixelScale;

const uchar kDashGray = 127;
const uchar kArrowGray = 226;
const uchar kSlotGray = 152;
const uchar kSlotGray1 = 151;

constexpr double kToRad = M_PI / 180.0;
constexpr double kToDeg = 180.0 / M_PI;

}  // end of namespace

// 将三维整数向量的x和y值组合成一个哈希值
// 用于avp地图空间网格索引
struct Vector3iHash {
  std::size_t operator()(const Eigen::Vector3i &pt) const {
    return (static_cast<size_t>(pt.x()) << 32) | pt.y();
  }
};

// 语义标签
enum SemanticLabel {
  kDashLine = 0,
  kArrowLine,
  kSlot,
  kTotalLabelNum  // flag that indicate label num
};

struct Slot {
  Eigen::Vector3d corners_[4];
};

// slot cornerPoints
class CornerPoint {
 public:
  CornerPoint() : CornerPoint({0, 0, 0}) {}
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
  Eigen::Vector3d center() const { return center_ / count_; }

 private:
  Eigen::Vector3d center_;
  unsigned int count_;
};
// image processing utils for slot detection
cv::Mat skeletonize(const cv::Mat &img);
void removeIsolatedPixels(cv::Mat &src, int min_neighbors = 1);
bool isSlotLongLine(const cv::Mat &line_img, const cv::Point2f &start,
                    const cv::Point2f &dir);
bool isSlotShortLine(const cv::Point2f &point1, const cv::Point2f &point2,
                     const cv::Mat &mask);

// IO method for data serialization and deserialization

// 要读取的文件内存格式布局
// 文件内容布局：
// +----------------+-------------------------------+
// | size (8 bytes) | element1 | element2 | ...     |
// +----------------+-------------------------------+
//         ↑
//         这里读取8字节到size变量

namespace io {
template <typename T>
// 数据加载函数
void loadVector(std::ifstream &ifs, std::vector<T> &vec) {
  // 读取Vector的大小
  size_t size{0};
  ifs.read(
    reinterpret_cast<char *>(&size),  
    sizeof(size));
  // 调整Vector容器的大小
  vec.resize(size);
  // 读取Vector的数据
  ifs.read(
    reinterpret_cast<char *>(vec.data()),  // 获取读取数据的指针
    size * sizeof(T));                     // 读取的字节数
}

template <typename T>
// 数据保存函数
void saveVector(std::ofstream &ofs, const std::vector<T> &vec) {
  // 定义Vector长度
  size_t size = vec.size();
  // 将Vector的长度写入文件
  ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
  // 将Vector的数据写入文件
  ofs.write(reinterpret_cast<const char *>(vec.data()), size * sizeof(T));
}
}  // namespace io