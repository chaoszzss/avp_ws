#pragma once
#include "primitives.h"
#include "slot.h"

// 地图类
class Map {
  // 车位点索引
  using SlotIndex = std::array<size_t, 4>;

 public:
  // 构造函数
  explicit Map() : semantic_grid_map_(SemanticLabel::kTotalLabelNum) {}

  // 新增方法获取地图尺寸和原点
  int getWidth() const { return (bounding_box_.max().x() - bounding_box_.min().x()) / kPixelScale; }  
  int getHeight() const { return (bounding_box_.max().y() - bounding_box_.min().y()) / kPixelScale; }
  Eigen::Vector3d getOrigin() const { return origin_; }

  // 获取地图边界框
  Eigen::AlignedBox3d getBoundingBox() const { return bounding_box_; }
  // 获取所有车位点
  std::vector<Slot> getAllSlots() const;
  // 获取语义元素
  const std::unordered_set<Eigen::Vector3i, Vector3iHash> &getSemanticElement(
      SemanticLabel label) const;
  const PrimitiveSet &primitives() const { return primitives_; }
  // 添加语义元素
  void addSemanticElement(SemanticLabel label, const Eigen::Vector3d &pt);
  // 添加车位点
  void addSlot(const Slot &new_slot);
  // 保存地图
  void save(const std::string &filename) const;
  // 加载地图
  void load(const std::string &filename);
  // 离散化线？
  void discretizeLine(const std::vector<Slot> &slots);
  void buildSemanticPrimitives();

 private:

  // int width_ = 1000;   // 根据实际地图尺寸设置
  // int height_ = 1000;
  Eigen::Vector3d origin_ = Eigen::Vector3d(-50, -50, 0); // 示例原点坐标

  // return the index of the corner point
  // 添加车位点角点
  size_t addSlotCorner(const Eigen::Vector3d &pt);

  std::vector<CornerPoint> corner_points_;  // slot corners
  std::vector<SlotIndex> slots_;  // index of slot points, 4 points for a slot
  std::vector<std::unordered_set<Eigen::Vector3i, Vector3iHash>>
      semantic_grid_map_;             // grid of semantic elements 语义元素网格
  Eigen::AlignedBox3d bounding_box_;  // bounding box of slot corners 车位点角点边界框
  PrimitiveSet primitives_;
};

// 获取绕Z轴旋转的Yaw角
// 将四元数转换为角度
template <typename T>
T GetYaw(const Eigen::Quaternion<T> &rotation) {
  const Eigen::Matrix<T, 3, 1> direction =
      rotation * Eigen::Matrix<T, 3, 1>::UnitX();
  return atan2(direction.y(), direction.x());
}
