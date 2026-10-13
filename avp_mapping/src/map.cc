#include <algorithm>

#include "avp_mapping/map.h"
#include "avp_mapping/semantic_downsampler.h"

namespace {

// 车位在当前帧被判为占用时，累积置信度每次增加的分值。
// 越大，新占用会更快冲过阈值。
constexpr float kScoreAlpha = 2.0f; // 1.0 -> 1.2 -> 1.5 -> 2.0
// 车位在当前帧被判为空闲时，置信度下降的幅度。
// 越大，系统越快把分数拉回空闲方向。
constexpr float kScoreBeta = 0.5f;
// 分数的饱和边界，仅用于防止数值发散。
constexpr float kScoreMin = -5.0f;
constexpr float kScoreMax = 5.0f;
// 分数达到或超过该值时，输出状态切换为“占用”。
constexpr float kScoreHighThreshold = 2.0f;
// 分数降到或低于该值时，输出状态切换为“空闲”。
constexpr float kScoreLowThreshold = -1.0f;
}  // namespace

/**
 * slots_ = [
  [10, 11, 12, 13],  // 停车位0的角点索引
  [20, 21, 22, 23],  // 停车位1的角点索引
  ...
]
 * corner_points_ = [
  {坐标数据0},  // 索引0
  {坐标数据1},  // 索引1
  ...
  {坐标数据10}, // 对应slots_[0][0]
  {坐标数据11}, // 对应slots_[0][1]
  ...
]
 */
std::vector<Slot> Map::getAllSlots() const 
{
  // 创建与slots_数量相同的空Slot对象
  std::vector<Slot> all_slots(slots_.size());
  // Slot：Eigen::Vector3d corners_[4];
  // 遍历每个停车位
  for (int i = 0; i < slots_.size(); ++i) {
    // 遍历停车位的4个角点
    for (int j = 0; j < 4; ++j) {
      // 关键索引操作：
      // 1. slots_[i][j] -> 获取第i个停车位的第j个角点的索引值
      // 2. corner_points_[索引] -> 获取对应的角点对象
      // 3. .center() -> 获取该角点的中心坐标，角点不是一个单独的点，是一片离散化后的区域
      all_slots[i].corners_[j] = corner_points_[slots_[i][j]].center();
    }
    // 设置占用状态
    all_slots[i].occupied = slot_occupied_[i];
    if (i < slot_scores_.size()) {
      all_slots[i].occupancy_score = slot_scores_[i];
    } else {
      all_slots[i].occupancy_score = 0.f;
    }
  }
  return std::move(all_slots);
}

const std::unordered_set<Eigen::Vector3i, Map::Vector3iHash>& Map::getSemanticElement(SemanticLabel label) const {
  return semantic_grid_map_.at(label);
}

// add semantic element
void Map::addSemanticElement(SemanticLabel label, const Eigen::Vector3d &pt) {
  Eigen::Vector3i grid_pt;
  grid_pt.x() = std::round(pt.x() * kPixelScaleInv);
  grid_pt.y() = std::round(pt.y() * kPixelScaleInv);
  grid_pt.z() = std::round(pt.z() * kPixelScaleInv);
  const auto &downsampler = avp_mapping::GetSemanticDownsampler();
  Eigen::Vector3i quantized = downsampler.Quantize(grid_pt);
  semantic_grid_map_[label].insert(quantized);
}

// add slot to map
size_t Map::addSlot(const Slot &new_slot, bool *is_new) {
  // add slot corner to map
  SlotIndex new_slot_index;
  Eigen::Vector3d new_slot_center = Eigen::Vector3d::Zero();
  for (int i = 0; i < 4; ++i) {
    new_slot_index[i] = addSlotCorner(new_slot.corners_[i]);
    new_slot_center += new_slot.corners_[i];
  }
  new_slot_center *= 0.25;
  // add to map if new slot
  bool isNewSlot = true;
  int found_index = -1;
  Eigen::Vector3d slot_center;
  for (int i = 0; i < slots_.size(); ++i) {
    const auto& slot_index = slots_[i];
    slot_center = Eigen::Vector3d::Zero();
    for (int j = 0; j < 4; ++j) {
      slot_center += corner_points_[slot_index[j]].center();
    }
    slot_center *= 0.25;
    if ((slot_center- new_slot_center).norm() < 1.0) {
      isNewSlot = false;
      found_index = i;
      break;
    }
  }
  if (isNewSlot) {
    slots_.push_back(new_slot_index);
    slot_occupied_.push_back(false);
    slot_scores_.push_back(0.f);
    if (is_new) {
      *is_new = true;
    }
    return slots_.size() - 1;
  } else {
    if (is_new) {
      *is_new = false;
    }
    return static_cast<size_t>(found_index);
  }
}

size_t Map::addSlotCorner(const Eigen::Vector3d &pt) {
  size_t index = corner_points_.size();
  for (int i = 0; i < corner_points_.size(); ++i) {
    if (corner_points_[i].absorb(pt, kNeighborDistance * kPixelScale)) {
      index = i;
      break;
    }
  }
  if (corner_points_.size() == index) {
    corner_points_.emplace_back(pt);
    bounding_box_.extend(pt);
  }
  return index;
}

bool Map::updateSlotOccupancy(size_t slot_index, bool observed_occupied) {
  if (slot_index >= slot_scores_.size()) {
    return observed_occupied;
  }
  float &score = slot_scores_[slot_index];
  score += observed_occupied ? kScoreAlpha : -kScoreBeta;
  score = std::max(kScoreMin, std::min(kScoreMax, score));

  const bool previous = slot_occupied_[slot_index];
  if (score >= kScoreHighThreshold) {
    slot_occupied_[slot_index] = true;
  } else if (score <= kScoreLowThreshold) {
    slot_occupied_[slot_index] = false;
  } else {
    slot_occupied_[slot_index] = previous;
  }
  return slot_occupied_[slot_index];
}

float Map::getSlotScore(size_t slot_index) const {
  if (slot_index < slot_scores_.size()) {
    return slot_scores_[slot_index];
  }
  return 0.f;
}

void Map::save(const std::string &filename) const {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()){
    std::cerr << "Unable to open map file: " << filename << std::endl;
    return;
  }

  io::saveVector(ofs, corner_points_);
  io::saveVector(ofs, slots_);
  // io::saveVector(ofs, slot_occupied_); // 添加这行
  std::cout << "save corners: " << corner_points_.size() << std::endl;
  std::cout << "save slots: " << slots_.size() << std::endl;

  std::vector<Eigen::Vector3i> semantic_grid;
  for (int i = 0; i < SemanticLabel::kTotalLabelNum; ++i) {
    semantic_grid.assign(semantic_grid_map_[i].begin(), semantic_grid_map_[i].end());
    io::saveVector(ofs, semantic_grid);
    std::cout << "save label_" << i << ": " << semantic_grid.size() << std::endl;
  }
  ofs.close();
  std::cout << "avp map saved to " << filename << std::endl;
}

void Map::load(const std::string &filename) {
  if (!corner_points_.empty()){
    std::cerr << "map have already loaded, fail to load map file: " << filename << std::endl;
    return;
  }
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()){
    std::cerr << "Unable to load map file: " << filename << std::endl;
    return;
  }
  io::loadVector(ifs, corner_points_);
  io::loadVector(ifs, slots_);
  // io::loadVector(ifs, slot_occupied_); // 添加这行
  for(const auto &corner : corner_points_){
    bounding_box_.extend(corner.center()); // extend bounding box
  }
  std::cout << "load corners: " << corner_points_.size() << std::endl;
  std::cout << "load slots: " << slots_.size() << std::endl;

  std::vector<Eigen::Vector3i> semantic_grid;
  for (int i = 0; i < SemanticLabel::kTotalLabelNum; ++i) {
    io::loadVector(ifs, semantic_grid);
    semantic_grid_map_[i].insert(semantic_grid.begin(), semantic_grid.end());
    std::cout << "load label_" << i << ": " << semantic_grid.size() << std::endl;
  }
  ifs.close();
  std::cout << "avp map loaded from " << filename << std::endl;

  slot_occupied_.assign(slots_.size(), false);
  slot_scores_.assign(slots_.size(), 0.f);
}
