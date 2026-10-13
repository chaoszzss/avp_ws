#include "../include/frame.h"

// 获取语义元素
const std::unordered_set<Eigen::Vector3i, Vector3iHash> &
Frame::getSemanticElement(SemanticLabel label) const {
  return semantic_grid_map_.at(label);
}

// 添加语义元素
// add semantic element
void Frame::addSemanticElement(SemanticLabel label, const Eigen::Vector3d &pt) {
  Eigen::Vector3i grid_pt;
  grid_pt.x() = std::round(pt.x() * kPixelScaleInv);
  grid_pt.y() = std::round(pt.y() * kPixelScaleInv);
  grid_pt.z() = std::round(pt.z() * kPixelScaleInv);
  semantic_grid_map_[label].insert(grid_pt);  // like a voxel filter
}

// 清除语义元素网格
void Frame::clearGridMap() {
  for (auto &it : semantic_grid_map_) {
    it.clear();
  }
  primitives_.clear();
}

// 保存primitives到CSV文件
void Frame::savePrimitives(const std::string& base_path, const std::string& suffix) const {
  // 创建输出文件名
  const std::string slot_filename = base_path + "_slot_points" + suffix + ".csv";
  const std::string dash_filename = base_path + "_dash_points" + suffix + ".csv";
  const std::string arrow_filename = base_path + "_arrow_points" + suffix + ".csv";
  const std::string slot_lines_filename = base_path + "_slot_lines" + suffix + ".csv";
  const std::string dash_lines_filename = base_path + "_dash_lines" + suffix + ".csv";
  const std::string arrow_dirs_filename = base_path + "_arrow_directions" + suffix + ".csv";

  // 保存slot points
  if (!primitives_.slot_points.empty()) {
    std::ofstream slot_file(slot_filename);
    slot_file << std::fixed << std::setprecision(6);
    slot_file << "x,y,type\n";
    for (const auto& pt : primitives_.slot_points) {
      slot_file << pt.x() << "," << pt.y() << ",slot_point\n";
    }
    slot_file.close();
  }

  // 保存dash points
  if (!primitives_.dash_points.empty()) {
    std::ofstream dash_file(dash_filename);
    dash_file << std::fixed << std::setprecision(6);
    dash_file << "x,y,type\n";
    for (const auto& pt : primitives_.dash_points) {
      dash_file << pt.x() << "," << pt.y() << ",dash_point\n";
    }
    dash_file.close();
  }

  // 保存arrow points
  if (!primitives_.arrow_points.empty()) {
    std::ofstream arrow_file(arrow_filename);
    arrow_file << std::fixed << std::setprecision(6);
    arrow_file << "x,y,type\n";
    for (const auto& pt : primitives_.arrow_points) {
      arrow_file << pt.x() << "," << pt.y() << ",arrow_point\n";
    }
    arrow_file.close();
  }

  // 保存slot lines
  if (!primitives_.slot_lines.empty()) {
    std::ofstream slot_lines_file(slot_lines_filename);
    slot_lines_file << std::fixed << std::setprecision(6);
    slot_lines_file << "center_x,center_y,direction_x,direction_y,length,type\n";
    for (const auto& line : primitives_.slot_lines) {
      slot_lines_file << line.center.x() << "," << line.center.y() << ","
                      << line.direction.x() << "," << line.direction.y() << ","
                      << line.length << ",slot_line\n";
    }
    slot_lines_file.close();
  }

  // 保存dash lines
  if (!primitives_.dash_lines.empty()) {
    std::ofstream dash_lines_file(dash_lines_filename);
    dash_lines_file << std::fixed << std::setprecision(6);
    dash_lines_file << "center_x,center_y,direction_x,direction_y,length,type\n";
    for (const auto& line : primitives_.dash_lines) {
      dash_lines_file << line.center.x() << "," << line.center.y() << ","
                      << line.direction.x() << "," << line.direction.y() << ","
                      << line.length << ",dash_line\n";
    }
    dash_lines_file.close();
  }

  // 保存arrow directions
  if (!primitives_.arrow_directions.empty()) {
    std::ofstream arrow_dirs_file(arrow_dirs_filename);
    arrow_dirs_file << std::fixed << std::setprecision(6);
    arrow_dirs_file << "pos_x,pos_y,dir_x,dir_y,type\n";
    for (const auto& arrow : primitives_.arrow_directions) {
      arrow_dirs_file << arrow.position.x() << "," << arrow.position.y() << ","
                      << arrow.direction.x() << "," << arrow.direction.y() << ","
                      << "arrow_direction\n";
    }
    arrow_dirs_file.close();
  }

  // 创建汇总文件
  const std::string summary_filename = base_path + "_summary" + suffix + ".csv";
  std::ofstream summary_file(summary_filename);
  summary_file << std::fixed << std::setprecision(6);
  summary_file << "type,count\n";
  summary_file << "slot_points," << primitives_.slot_points.size() << "\n";
  summary_file << "dash_points," << primitives_.dash_points.size() << "\n";
  summary_file << "arrow_points," << primitives_.arrow_points.size() << "\n";
  summary_file << "slot_lines," << primitives_.slot_lines.size() << "\n";
  summary_file << "dash_lines," << primitives_.dash_lines.size() << "\n";
  summary_file << "arrow_directions," << primitives_.arrow_directions.size() << "\n";
  summary_file << "total_points," << primitives_.totalPointCount() << "\n";
  summary_file.close();
}
