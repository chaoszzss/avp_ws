#pragma once
#include "primitives.h"
#include "slot.h"
#include <fstream>
#include <iomanip>

// 帧类 
class Frame {
 public:
  // 构造函数
  explicit Frame() : semantic_grid_map_(SemanticLabel::kTotalLabelNum) {}
  // 获取语义元素
  const std::unordered_set<Eigen::Vector3i, Vector3iHash>& getSemanticElement(
      SemanticLabel label) const;
  // 添加语义元素
  void addSemanticElement(SemanticLabel label, const Eigen::Vector3d& pt);
  // 清除语义元素网格
  void clearGridMap();
  // 获取语义元素网格
  std::vector<std::unordered_set<Eigen::Vector3i, Vector3iHash>>
  getSemanticGridMap() {
    return semantic_grid_map_;
  };

  PrimitiveSet& primitives() { return primitives_; }
  const PrimitiveSet& primitives() const { return primitives_; }

  // 保存primitives到CSV文件
  void savePrimitives(const std::string& base_path, const std::string& suffix = "") const;

  // 初始化位姿（世界）
  Eigen::Affine3d T_world_ipm_;  // init_pose
  // 更新位姿
  Eigen::Vector3d t_update; // for viewer
 private:
  // 语义元素网格
  std::vector<std::unordered_set<Eigen::Vector3i, Vector3iHash>>
      semantic_grid_map_;  // grid of semantic elements
  PrimitiveSet primitives_;
};
