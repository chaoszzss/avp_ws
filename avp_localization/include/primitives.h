#pragma once

#include <Eigen/Core>

#include <vector>

struct LinePrimitive {
  Eigen::Vector2d center{Eigen::Vector2d::Zero()};
  Eigen::Vector2d direction{Eigen::Vector2d::UnitX()};
  double length{0.0};
};

struct DirectedPointPrimitive {
  Eigen::Vector2d position{Eigen::Vector2d::Zero()};
  Eigen::Vector2d direction{Eigen::Vector2d::UnitX()};
};

struct PrimitiveSet {
  std::vector<Eigen::Vector2d> slot_points;
  std::vector<Eigen::Vector2d> dash_points;
  std::vector<Eigen::Vector2d> arrow_points;
  std::vector<LinePrimitive> slot_lines;
  std::vector<LinePrimitive> dash_lines;
  std::vector<DirectedPointPrimitive> arrow_directions;

  std::size_t totalPointCount() const {
    return slot_points.size() + dash_points.size() + arrow_points.size();
  }

  void clear() {
    slot_points.clear();
    dash_points.clear();
    arrow_points.clear();
    slot_lines.clear();
    dash_lines.clear();
    arrow_directions.clear();
  }
};
