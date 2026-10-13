#include "../include/avp_map.h"

std::vector<Slot> AvpMap::getAllSlots() const {
  std::vector<Slot> all_slots(slots_.size());
  for (int i = 0; i < slots_.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      all_slots[i].corners_[j] = corner_points_[slots_[i][j]].center();
    }
  }
  return std::move(all_slots);
}

const std::unordered_set<Eigen::Vector3i, Vector3iHash> &
AvpMap::getSemanticElement(SemanticLabel label) const {
  return semantic_grid_map_.at(label);
}

// add semantic element
void AvpMap::addSemanticElement(SemanticLabel label, const Eigen::Vector3d &pt) {
  Eigen::Vector3i grid_pt;
  grid_pt.x() = std::round(pt.x() * kPixelScaleInv);
  grid_pt.y() = std::round(pt.y() * kPixelScaleInv);
  grid_pt.z() = std::round(pt.z() * kPixelScaleInv);
  semantic_grid_map_[label].insert(grid_pt);  // like a voxel filter
}

// 向地图中添加车位元素
// add slot to map
void AvpMap::addSlot(const Slot &new_slot) {
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
  Eigen::Vector3d slot_center;
  for (const auto slot_index : slots_) {
    slot_center = Eigen::Vector3d::Zero();
    for (int j = 0; j < 4; ++j) {
      slot_center += corner_points_[slot_index[j]].center();
    }
    slot_center *= 0.25;
    if ((slot_center - new_slot_center).norm() < 1.0) {
      isNewSlot = false;
      break;
    }
  }
  if (isNewSlot) {
    slots_.push_back(new_slot_index);
  }
}

// 向地图中添加车位角点
size_t AvpMap::addSlotCorner(const Eigen::Vector3d &pt) {
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

// 保存地图
void AvpMap::save(const std::string &filename) const {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()) {
    std::cerr << "Unable to open map file: " << filename << std::endl;
    return;
  }

  io::saveVector(ofs, corner_points_);
  io::saveVector(ofs, slots_);
  std::cout << "save corners: " << corner_points_.size() << std::endl;
  std::cout << "save slots: " << slots_.size() << std::endl;

  std::vector<Eigen::Vector3i> semantic_grid;
  for (int i = 0; i < SemanticLabel::kTotalLabelNum; ++i) {
    semantic_grid.assign(semantic_grid_map_[i].begin(),
                         semantic_grid_map_[i].end());
    io::saveVector(ofs, semantic_grid);
    std::cout << "save label_" << i << ": " << semantic_grid.size()
              << std::endl;
  }
  ofs.close();
  std::cout << "avp map saved to " << filename << std::endl;
}

// 加载地图
void AvpMap::load(const std::string &filename) {
  if (!corner_points_.empty()) {
    std::cerr << "map have already loaded, fail to load map file: " << filename
              << std::endl;
    return;
  }
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Unable to load map file: " << filename << std::endl;
    return;
  }
  io::loadVector(ifs, corner_points_);
  io::loadVector(ifs, slots_);
  for (const auto &corner : corner_points_) {
    bounding_box_.extend(corner.center());  // extend bounding box
  }
  std::cout << "load corners: " << corner_points_.size() << std::endl;
  std::cout << "load slots: " << slots_.size() << std::endl;

  std::vector<Eigen::Vector3i> semantic_grid;
  for (int i = 0; i < SemanticLabel::kTotalLabelNum; ++i) {
    io::loadVector(ifs, semantic_grid);
    semantic_grid_map_[i].insert(semantic_grid.begin(), semantic_grid.end());
    std::cout << "load label_" << i << ": " << semantic_grid.size()
              << std::endl;
  }
  ifs.close();
  if (semantic_grid_map_[kSlot].empty()) {
    discretizeLine(getAllSlots());
  }

  std::cout << "avp map loaded from " << filename << std::endl;
}

// 多边形顶点进行线性插值，用于在相邻顶点之间插入新的点
// 1-2 2-3 3-0  没有处理0-1边，会让0-1边为空的边，用于路径规划
void interpolatePoints(std::vector<Eigen::Vector3d> &points) {
  if (points.size() < 2) points = {};

  std::vector<Eigen::Vector3d> result;
  for (size_t i = 0; i < points.size(); ++i) {
      if(i == 0)
          continue;
    const Eigen::Vector3d &p1 = points[i];
    const Eigen::Vector3d &p2 = points[(i + 1) % 4];

    result.push_back(p1);
    // 计算需要插入的点数（每隔0.05米插入一个点）
    int insertionsPerSegment =
        std::max(1, static_cast<int>((p1 - p2).norm() / 0.05));

    // 在相邻的点之间插入新的点
    for (int j = 1; j <= insertionsPerSegment; ++j) {
      double t = static_cast<double>(j) / (insertionsPerSegment + 1);
      double nx = p1(0) + t * (p2(0) - p1(0));
      double ny = p1(1) + t * (p2(1) - p1(1));
      double nz = p1(2) + t * (p2(2) - p1(2));
      result.emplace_back(nx, ny, nz);
    }
  }

  result.push_back(points.back());
  result.swap(points);
}

// 离散化车道线
void AvpMap::discretizeLine(const std::vector<Slot> &slots) {
  for (const auto &slot : slots) {
    std::vector<Eigen::Vector3d> single_slot;
    for (int j = 0; j < 4; ++j) {
      single_slot.push_back(slot.corners_[j]);
    }
    // 在相邻的车位角点之间插入新的点
    interpolatePoints(single_slot);
    for (const auto &pt : single_slot) {
      addSemanticElement(SemanticLabel::kSlot, pt);
    }
  }
  std::cout << "kSlot discretize points: "
            << getSemanticElement(SemanticLabel::kSlot).size() << std::endl;
}