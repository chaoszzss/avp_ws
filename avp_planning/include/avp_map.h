#pragma once
#include "slot.h"

struct TimedPose {
    double time_;
    Eigen::Vector3d t_;
    Eigen::Quaterniond R_;
};

class AvpMap {
  // 车位点索引：0-3
  using SlotIndex = std::array<size_t, 4>;

 public:
  // 初始化语义标签
  explicit AvpMap() : semantic_grid_map_(SemanticLabel::kTotalLabelNum) {}

  // 获取车位点包围盒
  Eigen::AlignedBox3d getBoundingBox() const { return bounding_box_; }
  // 获取所有的车位
  std::vector<Slot> getAllSlots() const;
  // 获取语义元素
  const std::unordered_set<Eigen::Vector3i, Vector3iHash> &getSemanticElement(
      SemanticLabel label) const;

  // 添加语义元素
  void addSemanticElement(SemanticLabel label, const Eigen::Vector3d &pt);
  // 添加车位
  void addSlot(const Slot &new_slot);

  // 保存地图
  void save(const std::string &filename) const;
  // 加载地图
  void load(const std::string &filename);
  // 离散化车道线
  void discretizeLine(const std::vector<Slot> &slots);

 private:
  // 添加车位角点
  // return the index of the corner point
  size_t addSlotCorner(const Eigen::Vector3d &pt);

  // 车位角点
  std::vector<CornerPoint> corner_points_;  // slot corners
  // 车位索引
  std::vector<SlotIndex> slots_;  // index of slot points, 4 points for a slot
  // 语义元素
  std::vector<std::unordered_set<Eigen::Vector3i, Vector3iHash>>
      semantic_grid_map_;             // grid of semantic elements
  // 车位点包围盒
  Eigen::AlignedBox3d bounding_box_;  // bounding box of slot corners
};

template <typename T>
T GetYaw(const Eigen::Quaternion<T> &rotation) {
  const Eigen::Matrix<T, 3, 1> direction =
      rotation * Eigen::Matrix<T, 3, 1>::UnitX();
  return atan2(direction.y(), direction.x());
}
