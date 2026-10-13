
#include "../include/KinematicModel.h"

/**
 * kinematic model
 * @param x pose x
 * @param y pose y
 * @param psi yaw angle
 * @param v velocity
 * @param l wheelbase
 * @param dt sample time period
 */
KinematicModel::KinematicModel(double x, double y, double psi, double v, double l, double dt) : x(x), y(y), psi(psi),
                                                                                                v(v), L(l), dt(dt) {}

/**
 * control input is acceleration and steering angle
 * @param velocity velocity  线速度
 * @param delta_f steering angle 转向角（注意：转向角delta_f和航向角yaw的区别）
 */
// 自行车模型的状态更新函数
// 注意：当采用外部定位信息时候，将不需要使用updateState函数来更新状态，而是直接使用定位信息
void KinematicModel::updateState(double velocity, double delta_f) {
    x = x + v * cos(psi) * dt;
    y = y + v * sin(psi) * dt;
    psi = psi + v / L * tan(delta_f) * dt;
    v = velocity;
}

/**
 * get state
 * @return
 */
vector<double> KinematicModel::getState() {
    return {x, y, psi, v};
}

/**
 * discrete state space ：状态空间的离散化
 * @param ref_delta  reference control input: 轨迹上参考点的转向角
 * @param ref_yaw  reference yaw：轨迹上参考点的航向角
 * @return
 */
// 将自行车模型的连续状态空间通过一阶欧拉法离散化
vector<MatrixXd> KinematicModel::stateSpace(double ref_delta, double ref_yaw) const {
    MatrixXd A(3, 3), B(3, 2);
    A << 1.0, 0.0, -v * dt * sin(ref_yaw),
         0.0, 1.0, v * dt * cos(ref_yaw),
         0.0, 0.0, 1.0;
    B << dt * cos(ref_yaw), 0,
         dt * sin(ref_yaw), 0,
         dt * tan(ref_delta) / L, v * dt / (L * cos(ref_delta) * cos(ref_delta));
    return {A, B};
}
