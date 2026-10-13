//
// Created by ubuntu on 25-3-8.
//
#include "../include/controller.h"
#include <iostream>
#include <algorithm>

/**
 * find the index of the point on the reference path which is closest to robot_state
 * @param robot_state robot state（x,y）
 * @param refer_path  reference path
 * @return the index of the closest point
 */
#define EPS 1.0e-4  // P和P_的差异阈值，当差异小于这个阈值时，认为P已经收敛了

// 计算目标点在参考路径上距离最短的路径点的下标
/**
 * @param robot_state 机器人状态：二维坐标点（至少）
 * @param refer_path 参考路径：二维坐标点构成的路径
 */
double Controller::calTargetIndex(const vector<double> &robot_state, const vector<vector<double>> &refer_path) {
    vector<double> dists;
    for (vector<double> xy: refer_path) {
        // 计算机器人当前的位置距离参考路径上每一个点的距离
        double dist = sqrt(pow(xy[0] - robot_state[0], 2) + pow(xy[1] - robot_state[1], 2));
        // 将路径点到机器人当前位置的距离存储到dist一维数组中
        dists.push_back(dist);
    }
    // 使用 min_element 查找 dists 中的最小值位置
    // 通过指针减法 min_element(...) - dists.begin() 得到索引
    return min_element(dists.begin(), dists.end()) - dists.begin(); //返回vector最小元素的下标
}

// 在基础版本上，引入前视距离l_d，确保目标点距离机器人足够远
// 用于纯追踪控制器
/**
 * @param robot_state 机器人状态：二维坐标点（至少）
 * @param refer_path 参考路径：二维坐标点构成的路径
 * @param l_d 前视距离（Look-ahead Distance），控制目标点的前瞻性
 * @return 满足前视距离条件的路径点索引
 */
double Controller::calTargetIndex(vector<double> robot_state, vector<vector<double>> refer_path, double l_d) {
    vector<double> dists;
    for (vector<double> xy: refer_path) {
        double dist = sqrt(pow(xy[0] - robot_state[0], 2) + pow(xy[1] - robot_state[1], 2));
        dists.push_back(dist);
    }

    // 找到初始最近点索引mid_ind
    double min_ind = min_element(dists.begin(), dists.end()) - dists.begin(); //返回vector最小元素的下标

    // 计算机器人到当前点的距离
    double delta_l = sqrt(
            pow(refer_path[min_ind][0] - robot_state[0], 2) + pow(refer_path[min_ind][1] - robot_state[1], 2));
    // 如果机器人距离当前点的距离小于前是距离l_d，则继续在路径上找
    while (l_d > delta_l && min_ind < refer_path.size() - 1) {
        delta_l = sqrt(pow(refer_path[min_ind + 1][0] - robot_state[0], 2) +
                       pow(refer_path[min_ind + 1][1] - robot_state[1], 2));
        min_ind += 1;
    }
    // 直到找到delta_l大于前视距离l_d时，返回当前索引
    return min_ind;
}

// 将角度规范到[-PI, PI]的范围中去
double normalizeAngle(double angle) {
    while (angle > PI) {
        angle -= 2.0 * PI;
    }
    while (angle < -PI) {
        angle += 2.0 * PI;
    }
    return angle;
}

// 计算黎卡提方程
/**
 * @param A 状态转移矩阵
 * @param B 控制矩阵
 * @param Q 状态权重矩阵:通常是对称半正定矩阵
 * @param R 控制权重矩阵：通常是对称正定矩阵
 * P：将 Qf 赋值给 P，作为 Riccati 方程的初始解
 * P_：用于存储迭代过程中计算的新解
 */
MatrixXd CalRicatti(MatrixXd A, MatrixXd B, MatrixXd Q, MatrixXd R) {
    MatrixXd Qf = Q;
    MatrixXd P = Qf;
    MatrixXd P_;
    for (int i = 0; i < 100; i++) {
        P_ = Q + A.transpose() * P * A -
             A.transpose() * P * B * (R + B.transpose() * P * B).inverse() * B.transpose() * P * A;
        // 检查 P_ 和 P 的差异是否小于一个很小的阈值 EPS:1e-4
        // 如果差异足够小，说明 P 已经收敛到 Riccati 方程的解，退出循环
        if ((P_ - P).maxCoeff() < EPS && (P - P_).maxCoeff() < EPS)break;
        // 否则，将 P_ 赋值给 P，继续迭代
        P = P_;
    }
    // P 是 LQR 控制问题中的关键矩阵，用于计算最优控制增益矩阵 K
    // K = （R + B^ * P * B）^(-1) * B^ * P * A
    // u = -K * x
    return P_;
}

// PID控制器
// 计算当前位置与参考路径的横向偏差，生成转向控制量（如方向盘转角），使车辆沿路径行驶
double Controller::PIDController(const vector<double> &robot_state, const vector<vector<double>> &refer_path,
                                 const KinematicModel &ugv) {
    double Kp = 0.7;
    double Ki = 0;
    double Kd = 10;
    static double sum_error = 0;    // 积分项累加器
    static double pre_error = 0;    // 上一次的误差（用于微分项）
    
    // 最小距离点的下标 
    double min_ind = calTargetIndex(robot_state, refer_path);
    // 从机器人当前位置指向最近路径点的方向角（全局坐标系）
    double alpha = atan2(refer_path[min_ind][1] - robot_state[1], refer_path[min_ind][0] - robot_state[0]);
    // 机器人到最近点的距离
    double l_d = sqrt(
            pow(refer_path[min_ind][0] - robot_state[0], 2) + pow(refer_path[min_ind][1] - robot_state[1], 2));
    // 横向角度误差，即目标方向角 alpha 与车辆当前航向角 ugv.psi 的差值
    double theta_e = alpha - ugv.psi;
    // 横向距离误差
    double e_y = -l_d * sin(theta_e);   

    /**
     * 参考路径:  [P0, P1, P2, ..., P_min, ...]
                 ↑ 机器人当前位置
                 |<-- l_d -->|
                 α (目标方向角)
        车辆航向: ψ
        航向误差: θ_e = α - ψ
        横向误差: e_y = -l_d * sin(θ_e)
     */

    // 希望将横向距离误差调整为0
    double error = 0 - e_y; 

    // PID控制器的输出并不是误差本身，而是根据误差的大小和变化率计算出的控制量，
    // 这里的控制量是转向角，所以需要将距离误差通过增益转换为角度量。
    // 比例项直接根据当前的误差大小产生控制量，如果误差是距离，而控制量是角度（必须搞清楚控制系统中能够用来控制的控制量），
    // 那么比例系数必须包含单位转换的因子。例如，每米的误差对应多少弧度的转向角调整。

    /**
     * 比例项 $K_p$：单位是 弧度/米。
       例如，$K_p = 0.7\ \text{rad/m}$ 表示每米横向误差需要补偿 $0.7\ \text{rad}$ 的转向角。
     
     * 微分项 $K_d$：单位是 弧度/(米/秒)。
       例如，$K_d = 10\ \text{rad/(m/s)}$ 表示横向误差变化率为 $1\ \text{m/s}$ 时，补偿 $10\ \text{rad}$ 的转向角。
     */

    // u：转向控制量（单位：弧度），被限制在 [-π/6, π/6] 范围内
    double u = error * Kp + sum_error * Ki + (error - pre_error) * Kd;
    // 将转向的控制量控制在[-PI/6, PI/6]范围内
    double delta_max = PI / 3;
    u = u > delta_max ? delta_max : u;
    u = u < -delta_max ? -delta_max : u;

    pre_error = error;   // 保存当前误差，用于下一次计算微分项
    sum_error += error;  // 累加误差，用于积分项（当前 Ki=0，实际未使用）

    return u;
}

// 纯追踪控制器：最后一小截距离控制器有一点小问题，就是在机器人到达路径点后，会出现来回转圈的情况？
double
Controller::PurePursuitController(const vector<double> &robot_state, const vector<vector<double>> &refer_path,
                                  const KinematicModel &ugv) {

    double l_d = 3; // 前视距离
    double L = 2; // 轴距

    // 获取最小距离点的下标
    double min_ind = calTargetIndex(robot_state, refer_path, l_d);
    // 从机器人当前位置指向最近路径点的方向角
    double alpha = atan2(refer_path[min_ind][1] - robot_state[1], refer_path[min_ind][0] - robot_state[0]);
    // 机器人到最近点的距离
    double l = sqrt(pow(refer_path[min_ind][0] - robot_state[0], 2) +
                    pow(refer_path[min_ind][1] - robot_state[1], 2));
    
    // 计算横向角度误差
    double e_y = normalizeAngle(alpha - ugv.psi);
    
    // 计算控制量
    double delta = atan(2 * L * sin(e_y) / l);
    // 控制量（归一化到[-PI, PI]的范围内）
    delta = normalizeAngle(delta);
    // 限制转角在[-PI / 3, PI / 3]范围内
    double delta_max = PI / 4;
    delta = delta > delta_max ? delta_max : delta;
    delta = delta < -delta_max ? -delta_max : delta;

    return delta;
}

// 斯坦利控制器:除了控制横向误差,还控制航向角误差
double Controller::StanlyController(const vector<double> &robot_state, const vector<vector<double>> &refer_path,
                                    const KinematicModel &ugv) {

    double k = 3.5; // 横向误差增益系数

    // 获取最小距离点的下标
    double min_ind = calTargetIndex(robot_state, refer_path);
    // 从机器人当前位置指向最近路径点的方向角（全局坐标系）
    double alpha = atan2(refer_path[min_ind][1] - robot_state[1], refer_path[min_ind][0] - robot_state[0]);
    // 机器人到最近点的距离
    double l_d = sqrt(pow(refer_path[min_ind][0] - robot_state[0], 2) +
                      pow(refer_path[min_ind][1] - robot_state[1], 2));
    // 计算横向距离误差
    double e_y = l_d * sin(alpha - ugv.psi);

    // 计算前进距离
    // 加入防0处理，防止除数为0
    double d = (ugv.v + EPS) / k;
    // 误差角
    double e_alpha = atan(e_y / d);

    // 航向角误差
    // 从参考路径点获取路径切线方向角（yaw）
    double path_yaw = refer_path[min_ind][2];
    double e_psi = normalizeAngle(path_yaw - ugv.psi);

    // 控制量（归一化到[-PI, PI]的范围内）
    double e_all = normalizeAngle(e_alpha + e_psi);
    // 限制总误差在[-PI/6, PI/6]的范围内
    e_all = e_all > PI / 3 ? PI / 3 :e_all;
    e_all = e_all < -PI / 3 ? -PI / 3 : e_all;

    return e_all;
}


// LQR控制器
// double Controller::LQRController(const vector<double> &robot_state, const vector<vector<double>> &refer_path,
//                                  const KinematicModel &ugv) {
//     // 1、定义状态空间维度
//     size_t state_dim = 3;
//     size_t control_dim = 2;

//     // 2、定义状态空间状态
//     // 获取参考点状态
//     double min_ind = calTargetIndex(robot_state, refer_path);
//     double ref_x = refer_path[min_ind][0];
//     double ref_y = refer_path[min_ind][1];
//     double ref_yaw = refer_path[min_ind][2];

//     // 获取当前机器人状态
//     double robot_x = robot_state[0];
//     double robot_y = robot_state[1];
//     double robot_yaw = robot_state[2];
//     double robot_v = robot_state[3];

//     // 定义状态误差向量：x_err = [Δx, Δy, Δpsi]
//     Vector3d state_error;
//     state_error << robot_x - ref_x,
//                    robot_y - ref_y,
//                    normalizeAngle(robot_yaw - ref_yaw);

//     // 通过路径曲率计算参考转向角ref_delta
//     double ref_delta = 0.0;
//     const double L = ugv.L; // 车辆轴距
//     if (min_ind > 0 && min_ind < refer_path.size() - 1) {
//         // 三点曲率计算：前一点、当前点、后一点
//         double x_prev = refer_path[min_ind - 1][0];
//         double y_prev = refer_path[min_ind - 1][1];
//         double x_curr = refer_path[min_ind][0];
//         double y_curr = refer_path[min_ind][1];
//         double x_next = refer_path[min_ind + 1][0];
//         double y_next = refer_path[min_ind + 1][1];

//         // 计算曲率半径
//         double dx_prev = x_curr - x_prev;
//         double dy_prev = y_curr - y_prev;
//         double dx_next = x_next - x_curr;
//         double dy_next = y_next - y_curr;

//         double d_theta = atan2(dy_next, dx_next) - atan2(dy_prev, dx_prev);
//         double ds_prev = sqrt(dx_prev * dx_prev + dy_prev * dy_prev);
//         double ds_next = sqrt(dx_next * dx_next + dy_next * dy_next);
//         double ds = (ds_prev + ds_next) / 2.0;

//         // 避免除以0
//         if (fabs(ds) > 1e-4 && fabs(d_theta) > 1e-4) {
//             double curvature = d_theta / ds; // 曲率 = 角度变化率 / 弧长变化
//             ref_delta = atan(L * curvature); // 转向角 = arctan(轴距 × 曲率)
//         }
//     }               

//     // 3、获取离散状态空间模型
//     // 这里的输入参数：参考点转向角、参考点航向角怎么选取？
//     vector<MatrixXd> state_space = ugv.stateSpace(ref_delta, ref_yaw);
//     MatrixXd A = state_space[0];
//     MatrixXd B = state_space[1];

//     // 4、定义状态权重矩阵Q和控制权重矩阵R
//     MatrixXd Q(state_dim, state_dim);
//     MatrixXd R(control_dim, control_dim);
//     Q << 1, 0, 0,   // x误差权重
//          0, 1, 0,   // y误差权重
//          0, 0, 1;   // 航向角权重
//     R << 1, 0,      // 转向角权重
//          0, 0;      // 加速度权重
    
//     // 5、计算Riccati方程
//     MatrixXd P = CalRicatti(A, B, Q, R);
    
//     // 6、计算最优控制增益矩阵 K
//     MatrixXd K = (R + B.transpose() * P * B).inverse() * B.transpose() * P * A;

//     // 计算控制量
//     VectorXd u = -K * state_error;
    
//     // 取出控制量中的转角
//     double delta = u(1);
    
//     // 对转角的范围进行限制
//     const double delte_max = PI / 4;
//     delta = delta > delte_max ? delte_max : delta;
//     delta = delta < -delte_max ? -delte_max : delta;

//     return delta;
// }

// LQR控制器
double Controller::LQRController(const vector<double> &robot_state, const vector<vector<double>> &refer_path,
                                 const KinematicModel &ugv) {
    // 1、定义状态空间维度
    size_t state_dim = 3;
    size_t control_dim = 2;

    // 2、定义状态空间状态
    // 获取参考点状态
    double min_ind = calTargetIndex(robot_state, refer_path);
    double ref_x = refer_path[min_ind][0];
    double ref_y = refer_path[min_ind][1];
    double ref_yaw = refer_path[min_ind][2];

    // 通过路径曲率计算参考转向角ref_delta
    double ref_delta = 0.0;
    const double L = ugv.L; // 车辆轴距
    if (min_ind > 0 && min_ind < refer_path.size() - 1) {
        // 三点曲率计算：前一点、当前点、后一点
        double x_prev = refer_path[min_ind - 1][0];
        double y_prev = refer_path[min_ind - 1][1];
        double x_curr = refer_path[min_ind][0];
        double y_curr = refer_path[min_ind][1];
        double x_next = refer_path[min_ind + 1][0];
        double y_next = refer_path[min_ind + 1][1];

        // 计算曲率半径
        double dx_prev = x_curr - x_prev;
        double dy_prev = y_curr - y_prev;
        double dx_next = x_next - x_curr;
        double dy_next = y_next - y_curr;

        double d_theta = atan2(dy_next, dx_next) - atan2(dy_prev, dx_prev);
        double ds_prev = sqrt(dx_prev * dx_prev + dy_prev * dy_prev);
        double ds_next = sqrt(dx_next * dx_next + dy_next * dy_next);
        double ds = (ds_prev + ds_next) / 2.0;

        // 避免除以0
        if (fabs(ds) > 1e-4 && fabs(d_theta) > 1e-4) {
            double curvature = d_theta / ds; // 曲率 = 角度变化率 / 弧长变化
            ref_delta = atan(L * curvature); // 转向角 = arctan(轴距 × 曲率)
        }
    }

    // 3、获取离散状态空间模型
    vector<MatrixXd> state_space = ugv.stateSpace(ref_delta, ref_yaw);
    MatrixXd A = state_space[0];
    MatrixXd B = state_space[1];

    // 4、定义状态权重矩阵Q和控制权重矩阵R（调整航向角权重）
    MatrixXd Q(state_dim, state_dim);
    MatrixXd R(control_dim, control_dim);
    Q << 10.0, 0.0, 0.0,   // x误差权重（增强位置跟踪）
          0.0, 10.0, 0.0,  // y误差权重
          0.0, 0.0, 50.0;  // 航向角权重（显著增强）
    R << 1.0, 0.0,      // 转向角权重
          0.0, 0.0;     // 加速度权重

    // 5、计算Riccati方程
    MatrixXd P = CalRicatti(A, B, Q, R);

    // 6、计算最优控制增益矩阵 K
    MatrixXd K = (R + B.transpose() * P * B).inverse() * B.transpose() * P * A;

    // 定义状态误差向量：x_err = [Δx, Δy, Δpsi]
    Vector3d state_error;
    state_error << robot_state[0] - ref_x,
                   robot_state[1] - ref_y,
                   normalizeAngle(robot_state[2] - ref_yaw);

    // 计算控制量
    VectorXd u = -K * state_error;

    // 取出控制量中的转角
    double delta = u(1);

    // 对转角的范围进行限制
    const double delta_max = PI / 3.0;
    delta = std::clamp(delta, -delta_max, delta_max);

    return delta;
}