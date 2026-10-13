#pragma once
#include <deque>

#include <Eigen/Geometry>
#include <Eigen/Core>

struct TimedPose {
  double time_;
  Eigen::Vector3d t_;
  Eigen::Quaterniond R_;
};

class PoseInterpolation {
 public:
  PoseInterpolation() = default;
  void Push(const TimedPose &pose);
  void TrimBefore(double time);
  double EarliestTime() const;
  bool LookUp(TimedPose &pose) const;

 private:
  std::deque<TimedPose> poses_;
};

template <typename T>
T GetYaw(const Eigen::Quaternion<T> &rotation) {
  const Eigen::Matrix<T, 3, 1> direction =
      rotation * Eigen::Matrix<T, 3, 1>::UnitX();
  return atan2(direction.y(), direction.x());
}
