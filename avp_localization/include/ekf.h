#pragma once
#include <deque>
#include <eigen3/Eigen/Geometry>
#include <random>
#include <cstddef>

// 基础噪声参数
// parameters for noise
constexpr double v_std = 0.7;            // 速度标准差:1e-2 -> 0.1 -> 0.2 -> 0.3 -> 0.4 -> 0.5 -> 0.7
constexpr double yaw_rate_std = 0.1;    // 偏航率标准差:1e-8 -> 0.01 -> 0.1

// 匹配参数
// parameters for match
constexpr double x_match_std = 0.01;       // 匹配x标准差:0.3 -> 0.1 -> 0.05 -> 0.03
constexpr double y_match_std = 0.01;       // 匹配y标准差:0.3 -> 0.1 -> 0.05 -> 0.03
constexpr double yaw_match_std = 5.0 / 180. * M_PI;  // 匹配偏航率标准差:10.0 -> 5.0 -> 3.0

// 自适应噪声参数
constexpr double base_v_noise = 0.1;      // 基础速度噪声
constexpr double base_yaw_noise = 0.01;   // 基础偏航噪声
constexpr double steering_noise_factor = 3.0; // 转向噪声系数
constexpr double low_speed_noise_factor = 2.0; // 低速噪声系数

// 车辆几何参数
constexpr double WHEELBASE = 2.7;         // 轴距 (米)
constexpr double MAX_STEERING_ANGLE = 0.6; // 最大转向角 (弧度)

// 增强状态结构
struct EnhancedState {
  double time;
  double x, y, yaw;           // 位置和姿态
  double velocity, yaw_rate;  // 速度和偏航率
  double sideslip;           // 侧偏角
  double steering_angle;     // 转向角

  // 6x6协方差矩阵 [x, y, yaw, v, yaw_rate, sideslip]
  Eigen::Matrix<double, 6, 6> P;
};

// 兼容性状态结构 (保持向后兼容)
struct State {
  double time;
  double x;
  double y;
  double yaw;
  Eigen::Matrix3d P; // 协方差矩阵

  // 转换到增强状态的构造函数
  operator EnhancedState() const {
    EnhancedState enhanced;
    enhanced.time = time;
    enhanced.x = x;
    enhanced.y = y;
    enhanced.yaw = yaw;
    enhanced.velocity = 0.0;
    enhanced.yaw_rate = 0.0;
    enhanced.sideslip = 0.0;
    enhanced.steering_angle = 0.0;
    enhanced.P.setZero();
    enhanced.P.block<3,3>(0,0) = P;
    return enhanced;
  }
};

// 自适应噪声模型类
class AdaptiveNoiseModel {
public:
  AdaptiveNoiseModel() = default;

  // 根据车辆状态更新噪声矩阵
  void updateMotionNoise(double velocity, double yaw_rate, double steering_angle);
  void updateObservationNoise(double fitness, double inlier_ratio);

  // 获取噪声矩阵
  Eigen::Matrix2d getMotionNoise() const { return Q_motion_; }
  Eigen::Matrix3d getObservationNoise() const { return R_observation_; }

private:
  // 速度相关噪声计算
  double velocityDependentNoise(double velocity) const;
  // 转向相关噪声计算
  double steeringDependentNoise(double steering_angle) const;
  // 基于匹配质量的观测噪声计算
  double qualityDependentNoise(double fitness, double inlier_ratio) const;

  Eigen::Matrix2d Q_motion_;      // 运动噪声矩阵 [v, yaw_rate]
  Eigen::Matrix3d R_observation_; // 观测噪声矩阵 [x, y, yaw]
};

// 状态插值类
// 用于管理和插值车辆状态历史记录
class StateInterpolation {
 public:
  StateInterpolation() = default;     // 默认构造函数
  void Push(const State &pose);       // 添加状态
  void TrimBefore(double time);       // 修剪之前的状态
  double EarliestTime() const;        // 最早的时间
  bool LookUp(State &pose) const;     // 查找状态

 private:
  std::deque<State> poses_;           // 状态队列
};

// EKF类
// 实现扩展卡尔曼滤波器算法 (增强版本)
class EKF {
 public:
  // 默认构造函数
  EKF() = default;
  // 构造函数 (兼容性接口)
  EKF(double time, double x, double y, double yaw);

  // === 原有接口 (保持向后兼容) ===
  // 预测 (原有接口)
  void ekfPredict(const double &time, const double &velocity,
                  const double &yaw_rate);
  // 更新 (原有接口)
  void ekfUpdate(const double &m_x, const double &m_y, const double &m_yaw,
                 double fitness, double inlier_ratio, bool measurement_valid);

  // === 新增强接口 ===
  // 增强预测 (包含转向角信息)
  void enhancedEkfPredict(const double &time, const double &velocity,
                         const double &yaw_rate, const double &steering_angle = 0.0);
  // 增强更新 (自适应噪声)
  void enhancedEkfUpdate(const double &m_x, const double &m_y, const double &m_yaw,
                        double fitness, double inlier_ratio, bool measurement_valid);

  // === 访问器 ===
  // 获取兼容性状态
  State getState() const;
  // 获取增强状态
  EnhancedState getEnhancedState() const { return enhanced_state_; }
  // 是否初始化
  bool isInit() const { return init_; }

  // DMA-EKF状态访问器
  double last_phi() const { return last_phi_; }
  double last_gamma() const { return last_gamma_; }
  double last_quality_score() const { return last_quality_score_; }

 private:
  // === 内部方法 ===
  // 初始化增强状态
  void initializeEnhancedState(double time, double x, double y, double yaw);

  // 运动学模型计算 (Ackermann转向模型)
  void computeKinematicModel(double dt, double velocity, double yaw_rate,
                            double steering_angle,
                            double &dx, double &dy, double &dyaw, double &dsideslip);

  // 计算雅可比矩阵
  void computeJacobian(double velocity, double yaw_rate, double steering_angle);

  // 角度归一化
  double normalizeAngle(double angle) const;

  // === 状态变量 ===
  bool init_{false};
  State state_;                    // 兼容性状态
  EnhancedState enhanced_state_;   // 增强状态
  StateInterpolation state_interpolation_;

  // === 噪声模型 ===
  AdaptiveNoiseModel adaptive_noise_;  // 自适应噪声模型

  // 原有噪声矩阵 (保持兼容性)
  Eigen::Matrix2d Qn;
  Eigen::Matrix3d Rn_match;

  // === 雅可比矩阵 ===
  Eigen::Matrix<double, 6, 6> F_;  // 状态转移雅可比
  Eigen::Matrix<double, 6, 2> G_;  // 控制输入雅可比
  Eigen::Matrix3d C_;              // 观测矩阵 (3x6，提取位置姿态)

  // === DMA-EKF参数 ===
  double rho_{0.95};              // 遗忘因子
  double alpha_{2.0};             // 自适应系数
  double quality_threshold_{0.5}; // 质量阈值
  double phi_cap_{5.0};           // 强跟踪上限
  double gamma_cap_{3.0};         // 观测降权上限
  Eigen::Matrix3d innovation_covariance_{Eigen::Matrix3d::Identity() * 0.1};

  // === 上次更新的状态指标 ===
  double last_phi_{1.0};
  double last_gamma_{1.0};
  double last_quality_score_{0.0};

  // === 增强模式控制 ===
  bool use_enhanced_model_{true};  // 是否使用增强运动学模型
  bool use_adaptive_noise_{true};  // 是否使用自适应噪声
};
