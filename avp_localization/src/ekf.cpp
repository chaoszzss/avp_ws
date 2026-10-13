#include "../include/ekf.h"
#include <cmath>
#include <iostream>
#include <limits>

// === 自适应噪声模型实现 ===

// 根据速度更新运动噪声
void AdaptiveNoiseModel::updateMotionNoise(double velocity, double yaw_rate, double steering_angle) {
  double v_noise = velocityDependentNoise(velocity);
  double yaw_noise = steeringDependentNoise(steering_angle);

  Q_motion_ << v_noise * v_noise, 0,
                0, yaw_noise * yaw_noise;
}

// 根据匹配质量更新观测噪声
void AdaptiveNoiseModel::updateObservationNoise(double fitness, double inlier_ratio) {
  double noise_factor = qualityDependentNoise(fitness, inlier_ratio);

  double x_noise = x_match_std * noise_factor;
  double y_noise = y_match_std * noise_factor;
  double yaw_noise = yaw_match_std * noise_factor;

  R_observation_ << x_noise * x_noise, 0, 0,
                   0, y_noise * y_noise, 0,
                   0, 0, yaw_noise * yaw_noise;
}

// 速度相关噪声计算
double AdaptiveNoiseModel::velocityDependentNoise(double velocity) const {
  // 低速时噪声增大，高速时噪声趋于稳定值
  return base_v_noise * (1.0 + low_speed_noise_factor * exp(-std::abs(velocity) / 2.0));
}

// 转向相关噪声计算
double AdaptiveNoiseModel::steeringDependentNoise(double steering_angle) const {
  // 转向时噪声增大
  return base_yaw_noise * (1.0 + steering_noise_factor * std::abs(steering_angle) / MAX_STEERING_ANGLE);
}

// 基于匹配质量的观测噪声计算
double AdaptiveNoiseModel::qualityDependentNoise(double fitness, double inlier_ratio) const {
  // 综合考虑fitness和inlier_ratio计算质量指标
  double quality = 0.6 * fitness + 0.4 * (1.0 - inlier_ratio);
  double threshold = 0.5; // 使用固定阈值

  // 质量越差，噪声越大
  if (quality > threshold) {
    return 1.0 + 2.0 * (quality - threshold);
  }
  return 1.0;
}

namespace {

// 线性插值函数：根据时间在两个状态之间进行线性插值
// 输入：起始状态，结束状态，时间
// 输出：插值后的状态
State interpolation(const State& start, const State& end,
                        double time) {
  // 计算插值因子 
  double factor = (time - start.time) / (end.time - start.time);
  // 插值后的状态
  State ret;
  ret.time = time;
  ret.x = start.x + factor * (end.x - start.x);
  ret.y = start.y + factor * (end.y - start.y);
  ret.yaw = start.yaw + factor * (end.yaw - start.yaw);
  ret.P = start.P + factor * (end.P - start.P);
  
  return ret;
}

}  // namespace

// 添加新的状态到历史记录
void StateInterpolation::Push(const State& pose) {
  // 如果时间晚于最早的时间，则添加到历史记录
  if (pose.time > EarliestTime()) {
    poses_.push_back(pose);
  }
}

// 清除早于指定时间的状态
void StateInterpolation::TrimBefore(double time) {
  // 如果时间早于最早的时间，则修剪
  while (!poses_.empty() && poses_.front().time < time) {
    poses_.pop_front();
  }
}

// 获取历史记录中最早的时间
double StateInterpolation::EarliestTime() const {
  return poses_.empty() ? 0. : poses_.front().time;
}

// 查找指定时间的状态
bool StateInterpolation::LookUp(State& pose) const {
  // 如果时间不在范围内，返回false
  if (poses_.empty() || pose.time < poses_.front().time ||
      pose.time > poses_.back().time) {
    if (poses_.empty()){
      std::cout << "poses empty!" << std::endl;
    }else{
      std::cout << std::to_string(pose.time) << ", poses have: " << std::to_string(poses_.front().time)
                << " -- " << std::to_string(poses_.back().time) << std::endl;
    }
    return false;
  }

  // 使用二分查找找到指定时间的状态
  auto end = std::lower_bound(
      poses_.begin(), poses_.end(), pose.time,
      [](const State& _pose, const double t) { return _pose.time < t; });
  
  // 如果end指向最后一个元素，则将end指向倒数第二个元素 
  if (end == poses_.end()) {
    end = std::prev(end);
  }

  // 如果在时间范围内，且end指向最后一个元素，则使用线性插值计算指定时间的状态
  pose = interpolation(*std::prev(end), *end, pose.time);
  return true;
}

// 初始化增强状态
void EKF::initializeEnhancedState(double time, double x, double y, double yaw) {
  enhanced_state_.time = time;
  enhanced_state_.x = x;
  enhanced_state_.y = y;
  enhanced_state_.yaw = yaw;
  enhanced_state_.velocity = 0.0;
  enhanced_state_.yaw_rate = 0.0;
  enhanced_state_.sideslip = 0.0;
  enhanced_state_.steering_angle = 0.0;

  // 初始化6x6协方差矩阵
  enhanced_state_.P.setIdentity();
  enhanced_state_.P.block<3,3>(0,0) = state_.P;  // 位置姿态部分使用原协方差
  enhanced_state_.P(3,3) = 1.0;  // 速度方差
  enhanced_state_.P(4,4) = 0.1;  // 偏航率方差
  enhanced_state_.P(5,5) = 0.01; // 侧偏角方差

  // 初始化雅可比矩阵
  F_.setIdentity();
  G_.setZero();
  C_.setZero();
  C_.block<3,3>(0,0) = Eigen::Matrix3d::Identity();  // 观测矩阵提取位置姿态

  // 初始化自适应噪声模型
  adaptive_noise_.updateMotionNoise(0.0, 0.0, 0.0);
  adaptive_noise_.updateObservationNoise(0.1, 1.0);
}

// 角度归一化
double EKF::normalizeAngle(double angle) const {
  while (angle > M_PI) angle -= 2 * M_PI;
  while (angle < -M_PI) angle += 2 * M_PI;
  return angle;
}

// 运动学模型计算 (Ackermann转向模型)
void EKF::computeKinematicModel(double dt, double velocity, double yaw_rate,
                               double steering_angle,
                               double &dx, double &dy, double &dyaw, double &dsideslip) {
  if (use_enhanced_model_) {
    // 使用Ackermann转向几何
    if (std::abs(steering_angle) > 1e-6) {
      // 转向半径计算
      double turning_radius = WHEELBASE / std::tan(std::abs(steering_angle));

      // 侧偏角估计
      dsideslip = std::atan(0.5 * std::tan(steering_angle));

      // 考虑侧偏的运动学
      double v_longitudinal = velocity * std::cos(dsideslip);
      double v_lateral = velocity * std::sin(dsideslip);

      dx = dt * (v_longitudinal * std::cos(enhanced_state_.yaw) -
                 v_lateral * std::sin(enhanced_state_.yaw));
      dy = dt * (v_longitudinal * std::sin(enhanced_state_.yaw) +
                 v_lateral * std::cos(enhanced_state_.yaw));
      dyaw = dt * velocity / WHEELBASE * std::sin(dsideslip);
    } else {
      // 直线运动
      dsideslip = 0.0;
      dx = dt * velocity * std::cos(enhanced_state_.yaw);
      dy = dt * velocity * std::sin(enhanced_state_.yaw);
      dyaw = dt * yaw_rate;
    }
  } else {
    // 使用原始简化模型 (保持兼容性)
    dsideslip = 0.0;
    dx = dt * velocity * std::cos(enhanced_state_.yaw + M_PI_2);
    dy = dt * velocity * std::sin(enhanced_state_.yaw + M_PI_2);
    dyaw = dt * yaw_rate;
  }
}

// 计算雅可比矩阵
void EKF::computeJacobian(double velocity, double yaw_rate, double steering_angle) {
  double dt = 0.1; // 假设时间步长，实际应从外部传入

  if (use_enhanced_model_) {
    // 状态转移雅可比 F (6x6)
    F_.setIdentity();

    if (std::abs(steering_angle) > 1e-6) {
      double sideslip = std::atan(0.5 * std::tan(steering_angle));
      double v_long = velocity * std::cos(sideslip);
      double v_lat = velocity * std::sin(sideslip);

      // 位置对角度的偏导
      F_(0,2) = dt * (-v_long * std::sin(enhanced_state_.yaw) - v_lat * std::cos(enhanced_state_.yaw));
      F_(1,2) = dt * (v_long * std::cos(enhanced_state_.yaw) - v_lat * std::sin(enhanced_state_.yaw));

      // 位置对侧偏角的偏导
      F_(0,5) = dt * (-velocity * std::sin(sideslip) * std::cos(enhanced_state_.yaw) -
                       velocity * std::cos(sideslip) * std::sin(enhanced_state_.yaw));
      F_(1,5) = dt * (-velocity * std::sin(sideslip) * std::sin(enhanced_state_.yaw) +
                       velocity * std::cos(sideslip) * std::cos(enhanced_state_.yaw));

      // 角度对侧偏角的偏导
      F_(2,5) = dt * velocity / WHEELBASE * std::cos(sideslip);
    }

    // 控制输入雅可比 G (6x2)
    G_.setZero();
    if (std::abs(steering_angle) > 1e-6) {
      double sideslip = std::atan(0.5 * std::tan(steering_angle));
      G_(0,0) = dt * std::cos(sideslip) * std::cos(enhanced_state_.yaw) -
                 dt * std::sin(sideslip) * std::sin(enhanced_state_.yaw);
      G_(1,0) = dt * std::cos(sideslip) * std::sin(enhanced_state_.yaw) +
                 dt * std::sin(sideslip) * std::cos(enhanced_state_.yaw);
      G_(2,1) = dt;
    } else {
      G_(0,0) = dt * std::cos(enhanced_state_.yaw);
      G_(1,0) = dt * std::sin(enhanced_state_.yaw);
      G_(2,1) = dt;
    }
  } else {
    // 原始雅可比 (兼容模式)
    F_.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    F_(0,2) = -dt * velocity * std::sin(enhanced_state_.yaw + M_PI_2);
    F_(1,2) = dt * velocity * std::cos(enhanced_state_.yaw + M_PI_2);

    G_.block<3,2>(0,0) << dt * std::cos(enhanced_state_.yaw + M_PI_2), 0,
                           dt * std::sin(enhanced_state_.yaw + M_PI_2), 0,
                           0, dt;
  }
}

// 初始化EKF的状态 (增强版本)
EKF::EKF(double time, double x, double y, double yaw)
{
    // 初始化传统噪声协方差矩阵 (兼容性)
    Qn << v_std * v_std, 0,
        0, yaw_rate_std * yaw_rate_std;

    Rn_match << x_match_std * x_match_std, 0, 0,
        0, y_match_std * y_match_std, 0,
        0, 0, yaw_match_std * yaw_match_std;

    // 初始化兼容性状态
    state_.x = x;
    state_.y = y;
    state_.yaw = yaw;
    state_.P = 1.0 * Eigen::Matrix3d::Identity();
    state_.time = time;

    // 初始化增强状态 (暂时禁用增强模型)
    // initializeEnhancedState(time, x, y, yaw);
    enhanced_state_.x = x;
    enhanced_state_.y = y;
    enhanced_state_.yaw = yaw;
    enhanced_state_.velocity = 0.0;
    enhanced_state_.yaw_rate = 0.0;
    enhanced_state_.sideslip = 0.0;
    enhanced_state_.steering_angle = 0.0;
    enhanced_state_.time = time;
    enhanced_state_.P.setIdentity();
    enhanced_state_.P.block<3,3>(0,0) = state_.P;

    init_ = true;
    state_interpolation_.Push(state_);
}

// EKF预测 (保持原有稳定运动学模型，添加安全改进)
void EKF::ekfPredict(const double &time, const double &velocity, const double &yaw_rate)
{
    // 计算当前时间与上次预测时间的时间差
    double dt = time - state_.time;
    // 如果时间差为0，则不进行预测
    if (dt == 0)
        return;

    // car heading y-axis in vehicle frame,velocity direction is y-axis
    // 保持原有运动学模型不变，确保稳定性
    Eigen::Matrix3d A;
    A << 0, 0, -velocity * sin(state_.yaw + M_PI_2),
        0, 0, velocity * cos(state_.yaw + M_PI_2),
        0, 0, 0;
    // 状态转移矩阵
    Eigen::Matrix<double, 3, 2> U;
    U << cos(state_.yaw), 0,
        sin(state_.yaw), 0,
        0, 1;

    // 状态转移矩阵
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + dt * A;
    // 控制矩阵
    Eigen::Matrix<double, 3, 2> V;
    V = dt * U;

    // 使用运动学模型预测下一状态 (保持原有公式)
    state_.x += dt * velocity * cos(state_.yaw + M_PI_2);
    state_.y += dt * velocity * sin(state_.yaw + M_PI_2);
    state_.yaw += dt * yaw_rate;
    // 对yaw角度进行归一化，保持在[-π, π]范围内
    if (state_.yaw > M_PI)
    {
        state_.yaw -= 2 * M_PI;
    }
    // 如果角度小于-180度，则加上360度
    if (state_.yaw < -M_PI)
    {
        state_.yaw += 2 * M_PI;
    }

    // === 安全改进：根据速度自适应调整运动噪声 ===
    Eigen::Matrix2d Q_adj = Qn;
    if (std::abs(velocity) < 0.5) {
        // 低速时增加运动噪声
        double factor = 1.0 + 2.0 * exp(-std::abs(velocity) / 0.2);
        Q_adj = factor * Qn;
    }

    // 运动状态协方差矩阵计算
    state_.P = F * state_.P * F.transpose() + V * Q_adj * V.transpose();
    state_.time = time;
    // 将预测状态添加到插值对象
    state_interpolation_.Push(state_);

    // 同步增强状态
    enhanced_state_.x = state_.x;
    enhanced_state_.y = state_.y;
    enhanced_state_.yaw = state_.yaw;
    enhanced_state_.velocity = velocity;
    enhanced_state_.yaw_rate = yaw_rate;
    enhanced_state_.time = time;
    enhanced_state_.P.block<3,3>(0,0) = state_.P;
}

// EKF更新 (添加安全的自适应噪声改进)
void EKF::ekfUpdate(const double &m_x, const double &m_y, const double &m_yaw,
                    double fitness, double inlier_ratio, bool measurement_valid) {
  if (!init_) {
    return;
  }

  // 观测矩阵
  const Eigen::Matrix3d C = Eigen::Matrix3d::Identity();

  // 观测残差计算（观测值与预测值的差值）
  Eigen::Vector3d residual;
  residual(0) = m_x - state_.x;
  residual(1) = m_y - state_.y;
  residual(2) = m_yaw - state_.yaw;
  // 对yaw角度进行归一化，保持在[-π, π]范围内
  if (residual(2) > M_PI) {
    residual(2) -= 2 * M_PI;
  }
  if (residual(2) < -M_PI) {
    residual(2) += 2 * M_PI;
  }

  if (!measurement_valid) {
    // 无法信任的观测直接跳过更新，仅记录自适应指标便于诊断
    innovation_covariance_ = rho_ * innovation_covariance_ +
                             (1.0 - rho_) * (residual * residual.transpose());
    last_phi_ = 1.0;
    last_gamma_ = gamma_cap_;
    last_quality_score_ = std::numeric_limits<double>::infinity();
    return;
  }

  Eigen::Matrix3d P_base = state_.P;
  Eigen::Matrix3d S_base = P_base + Rn_match;

  // 新息协方差的指数加权估计
  innovation_covariance_ = rho_ * innovation_covariance_ +
                           (1.0 - rho_) * (residual * residual.transpose());

  double trace_s_base = S_base.trace();
  double phi = 1.0;
  if (trace_s_base > 1e-9) {
    phi = std::max(1.0, innovation_covariance_.trace() / trace_s_base);
  }
  phi = std::min(phi, phi_cap_);
  Eigen::Matrix3d P_pred = phi * P_base;

  double quality = 0.6 * fitness + 0.4 * (1.0 - inlier_ratio);
  if (!std::isfinite(quality)) {
    quality = quality_threshold_;
  }
  double gamma = 1.0;
  if (quality > quality_threshold_) {
    gamma = 1.0 + alpha_ * (quality - quality_threshold_);
  }
  gamma = std::min(std::max(1.0, gamma), gamma_cap_);

  // === 安全改进：基于质量的观测噪声调整 ===
  Eigen::Matrix3d R_adj = Rn_match;
  if (quality > quality_threshold_) {
    double noise_factor = 1.0 + std::min(2.0, (quality - quality_threshold_));
    R_adj = noise_factor * Rn_match;
  }

  Eigen::Matrix3d S_adj = P_pred + R_adj;
  Eigen::Matrix3d K = P_pred * C.transpose() * S_adj.inverse();

  Eigen::Vector3d correct = K * residual;
  // 更新状态
  state_.x += correct(0);
  state_.y += correct(1);
  state_.yaw += correct(2);
  // 归一化yaw角
  if (state_.yaw > M_PI) {
    state_.yaw -= 2 * M_PI;
  }
  if (state_.yaw < -M_PI) {
    state_.yaw += 2 * M_PI;
  }

  // Joseph形式更新协方差以提高数值稳定性
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d temp = I - K * C;
  state_.P = temp * P_pred * temp.transpose() + K * R_adj * K.transpose();

  last_phi_ = phi;
  last_gamma_ = gamma;
  last_quality_score_ = quality;
  printf("Safe Update t = %.4f, x: %.3f, y: %.3f, yaw: %.2f, phi: %.2f, gamma: %.2f, Q: %.2f, inliers: %.1f%%, fitness: %.4f \n",
         state_.time, state_.x, state_.y, state_.yaw, last_phi_, last_gamma_,
         last_quality_score_, inlier_ratio * 100.0, fitness);

  // 同步增强状态
  enhanced_state_.x = state_.x;
  enhanced_state_.y = state_.y;
  enhanced_state_.yaw = state_.yaw;
  enhanced_state_.P.block<3,3>(0,0) = state_.P;
}

// === 新增强方法实现 ===

// 获取兼容性状态
State EKF::getState() const {
  State compat_state;
  compat_state.time = enhanced_state_.time;
  compat_state.x = enhanced_state_.x;
  compat_state.y = enhanced_state_.y;
  compat_state.yaw = enhanced_state_.yaw;
  compat_state.P = enhanced_state_.P.block<3,3>(0,0);
  return compat_state;
}

// 增强预测 (包含转向角信息)
void EKF::enhancedEkfPredict(const double &time, const double &velocity,
                             const double &yaw_rate, const double &steering_angle) {
  if (!init_) return;

  double dt = time - enhanced_state_.time;
  if (dt <= 0) return;

  // 更新自适应噪声模型
  if (use_adaptive_noise_) {
    adaptive_noise_.updateMotionNoise(velocity, yaw_rate, steering_angle);
  }

  // 计算运动学模型
  double dx, dy, dyaw, dsideslip;
  computeKinematicModel(dt, velocity, yaw_rate, steering_angle,
                       dx, dy, dyaw, dsideslip);

  // 状态预测
  enhanced_state_.x += dx;
  enhanced_state_.y += dy;
  enhanced_state_.yaw += dyaw;
  enhanced_state_.velocity = velocity;
  enhanced_state_.yaw_rate = yaw_rate;
  enhanced_state_.sideslip += dsideslip;
  enhanced_state_.steering_angle = steering_angle;
  enhanced_state_.time = time;

  // 角度归一化
  enhanced_state_.yaw = normalizeAngle(enhanced_state_.yaw);

  // 计算雅可比矩阵
  computeJacobian(velocity, yaw_rate, steering_angle);

  // 协方差预测
  Eigen::Matrix2d Q = use_adaptive_noise_ ? adaptive_noise_.getMotionNoise() : Qn;
  enhanced_state_.P = F_ * enhanced_state_.P * F_.transpose() + G_ * Q * G_.transpose();

  // 同步兼容性状态
  state_.x = enhanced_state_.x;
  state_.y = enhanced_state_.y;
  state_.yaw = enhanced_state_.yaw;
  state_.time = enhanced_state_.time;
  state_.P = enhanced_state_.P.block<3,3>(0,0);
  state_interpolation_.Push(state_);
}

// 增强更新 (自适应噪声)
void EKF::enhancedEkfUpdate(const double &m_x, const double &m_y, const double &m_yaw,
                           double fitness, double inlier_ratio, bool measurement_valid) {
  if (!init_) return;

  // 更新自适应观测噪声
  if (use_adaptive_noise_) {
    adaptive_noise_.updateObservationNoise(fitness, inlier_ratio);
  }

  // 观测向量
  Eigen::Vector3d observation(m_x, m_y, m_yaw);

  // 观测预测
  Eigen::Vector3d predicted_obs = C_ * Eigen::Vector3d(enhanced_state_.x, enhanced_state_.y, enhanced_state_.yaw);

  // 新息计算
  Eigen::Vector3d residual = observation - predicted_obs;
  residual(2) = normalizeAngle(residual(2));

  // 自适应观测噪声
  Eigen::Matrix3d R = use_adaptive_noise_ ? adaptive_noise_.getObservationNoise() : Rn_match;

  if (!measurement_valid) {
    // 无效观测，仅更新自适应指标
    innovation_covariance_ = rho_ * innovation_covariance_ +
                             (1.0 - rho_) * (residual * residual.transpose());
    last_phi_ = 1.0;
    last_gamma_ = gamma_cap_;
    last_quality_score_ = std::numeric_limits<double>::infinity();
    return;
  }

  // 计算自适应参数
  Eigen::Matrix<double, 6, 6> P_base = enhanced_state_.P;
  Eigen::Matrix3d S_base = C_ * P_base.block<3,3>(0,0) * C_.transpose() + R;

  // 新息协方差估计
  innovation_covariance_ = rho_ * innovation_covariance_ +
                           (1.0 - rho_) * (residual * residual.transpose());

  // 强跟踪因子
  double trace_s_base = S_base.trace();
  double phi = 1.0;
  if (trace_s_base > 1e-9) {
    phi = std::max(1.0, innovation_covariance_.trace() / trace_s_base);
  }
  phi = std::min(phi, phi_cap_);

  // 观测降权因子
  double quality = 0.6 * fitness + 0.4 * (1.0 - inlier_ratio);
  if (!std::isfinite(quality)) {
    quality = quality_threshold_;
  }
  double gamma = 1.0;
  if (quality > quality_threshold_) {
    gamma = 1.0 + alpha_ * (quality - quality_threshold_);
  }
  gamma = std::min(std::max(1.0, gamma), gamma_cap_);

  // 自适应调整
  Eigen::Matrix<double, 6, 6> P_pred = P_base;
  P_pred.block<3,3>(0,0) = phi * P_base.block<3,3>(0,0);
  Eigen::Matrix3d R_adj = gamma * R;

  // 卡尔曼增益计算
  Eigen::Matrix3d S_adj = C_ * P_pred.block<3,3>(0,0) * C_.transpose() + R_adj;
  Eigen::Matrix<double, 6, 3> K;
  K.block<3,3>(0,0) = P_pred.block<3,3>(0,0) * C_.transpose() * S_adj.inverse();

  // 状态更新
  Eigen::Matrix<double, 6, 1> correction = K * residual;
  enhanced_state_.x += correction(0);
  enhanced_state_.y += correction(1);
  enhanced_state_.yaw += correction(2);
  enhanced_state_.velocity += correction(3);
  enhanced_state_.yaw_rate += correction(4);
  enhanced_state_.sideslip += correction(5);

  // 角度归一化
  enhanced_state_.yaw = normalizeAngle(enhanced_state_.yaw);

  // Joseph形式协方差更新 (只更新位置姿态部分)
  Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d temp3 = I3 - K.block<3,3>(0,0) * C_;
  enhanced_state_.P.block<3,3>(0,0) = temp3 * P_pred.block<3,3>(0,0) * temp3.transpose() +
                                    K.block<3,3>(0,0) * R_adj * K.block<3,3>(0,0).transpose();

  // 更新指标
  last_phi_ = phi;
  last_gamma_ = gamma;
  last_quality_score_ = quality;

  // 同步兼容性状态
  state_.x = enhanced_state_.x;
  state_.y = enhanced_state_.y;
  state_.yaw = enhanced_state_.yaw;
  state_.time = enhanced_state_.time;
  state_.P = enhanced_state_.P.block<3,3>(0,0);

  printf("Enhanced Update t = %.4f, x: %.3f, y: %.3f, yaw: %.2f, phi: %.2f, gamma: %.2f, Q: %.2f, inliers: %.1f%%, fitness: %.4f \n",
         enhanced_state_.time, enhanced_state_.x, enhanced_state_.y, enhanced_state_.yaw,
         last_phi_, last_gamma_, last_quality_score_, inlier_ratio * 100.0, fitness);
}
