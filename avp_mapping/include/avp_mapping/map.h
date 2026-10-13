#pragma once
#include "avp_mapping/slot.h"

class Map {
  using SlotIndex = std::array<size_t, 4>;
  struct Vector3iHash {
    std::size_t operator()(const Eigen::Vector3i &pt) const {
      return (static_cast<size_t>(pt.x()) << 32) | pt.y();
    }
  };
 public:
  explicit Map() : semantic_grid_map_(SemanticLabel::kTotalLabelNum) { }

  Eigen::AlignedBox3d getBoundingBox() const { return bounding_box_; }
  std::vector<Slot> getAllSlots() const;
  const std::unordered_set<Eigen::Vector3i, Vector3iHash>& getSemanticElement(SemanticLabel label) const;

  void addSemanticElement(SemanticLabel label, const Eigen::Vector3d &pt);
  size_t addSlot(const Slot &new_slot, bool *is_new = nullptr);
  bool updateSlotOccupancy(size_t slot_index, bool observed_occupied);
  float getSlotScore(size_t slot_index) const;

  void save(const std::string &filename) const;
  void load(const std::string &filename);
 private:
  // return the index of the corner point
  size_t addSlotCorner(const Eigen::Vector3d &pt);

  std::vector<CornerPoint> corner_points_;  // slot corners
  std::vector<SlotIndex> slots_; // index of slot points, 4 points for a slot
  std::vector<bool> slot_occupied_; // 存储每个车位的占用状态
  std::vector<float> slot_scores_; // 车位占用置信度
  std::vector<std::unordered_set<Eigen::Vector3i, Vector3iHash>> semantic_grid_map_;  // grid of semantic elements
  Eigen::AlignedBox3d bounding_box_; // bounding box of slot corners
};
