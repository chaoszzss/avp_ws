
#ifndef KINEMATICMODEL_H
#define KINEMATICMODEL_H

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class KinematicModel {
public:
    // x:pose_x, y:pose_y, psi:yaw angle, v:velocity, L:wheelbase, dt:sample time period
    double x, y, psi, v, L, dt;
public:
    // 默认构造函数
    KinematicModel();

    // 带参数的构造函数
    KinematicModel(double x, double y, double psi, double v, double l, double dt);

    // 获取状态
    vector<double> getState();

    // 更新状态
    void updateState(double velocity, double delta_f);

    // 将连续状态空间离散化
    vector<MatrixXd> stateSpace(double ref_delta, double ref_yaw) const;

};


#endif
