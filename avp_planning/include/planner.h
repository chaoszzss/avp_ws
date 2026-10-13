//
// Created by ubuntu on 25-3-2.
//

#ifndef AVP_WS_PLANNER_H
#define AVP_WS_PLANNER_H

#endif //AVP_WS_PLANNER_H

#include <iostream>
#include <cmath>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>  // reverse
#include <Eigen/Dense>

// 新增头文件（需在planner.h中声明）
#include <reeds_shepp.h>
#include <memory>

using namespace std;
using namespace Eigen;

// 功能：为Vector2i类型（二维整数向量）生成哈希值
// 哈希容器（unordered_set/unordered_map）需要通过哈希值快速定位元素
// C++标准库没有为自定义类型（如Eigen的Vector2i）提供默认哈希实现
// 哈希表查找时间复杂度为O(1)，比遍历查找快100倍以上
// 在Hybrid A这种需要频繁状态检查的算法中至关重要
struct Vector2iHash {
    size_t operator()(const Vector2i &v) const {
        size_t h1 = hash<int>{}(v.x());
        size_t h2 = hash<int>{}(v.y());
        return h1 ^ (h2 << 1);
    }
};

// 修改 close_table 的键类型为位置+角度离散化
struct StateKey {
    Vector2i pos;
    int theta_quantized;
    
    StateKey(const Vector2i& p, double theta) 
        : pos(p), theta_quantized(static_cast<int>(theta / (M_PI / 36))) {}
    
    // 必须定义 operator==
    bool operator==(const StateKey& other) const {
        return pos == other.pos && 
               theta_quantized == other.theta_quantized;
    }
};

struct StateKeyHash {
    size_t operator()(const StateKey& k) const {
        return Vector2iHash()(k.pos) ^ hash<int>()(k.theta_quantized);
    }
};

bool isNearObstacle(const Vector2i& pos, const unordered_set<Vector2i, Vector2iHash>& obstacles_index, double safety_distance);

void computeBSplineBasis(double t, int degree, const std::vector<double>& knots, Eigen::VectorXd& basis);

/**
 * 补充：带参数的构造函数和不带参数的构造函数区别
 * 构造函数的作用：初始化对象
 * 带参构造：强制要求有效状态（如路径节点必须有坐标）
 * 无参构造：允许延迟初始化（如规划器参数可后续配置）
 */

// 定义车辆状态
struct State {
    double x,            // x坐标
    y,                   // y坐标
    theta;               // 朝向角
    double g;           // 累计代价
    double h;          // 启发式代价
    double f;           // 总估计代价
    State* parent;      // 父节点指针
    bool is_backward;     // 新增运动方向标识

    // 构造函数
    // 用于创建搜索树中的的新节点
    State(double x, double y, double theta, double g, double h, State* parent, bool is_backward = false)
            : x(x), y(y), theta(theta), g(g), h(h), f(g + h), parent(parent), is_backward(is_backward) {}

    // 比较器，用于优先队列
    bool operator>(const State& other) const {
        return f > other.f;
    }
};

// 优先队列的比较器（按f值升序排列）
struct CompareState {
    bool operator()(const State* a, const State* b) {
        return a->f > b->f; // 小顶堆，优先处理 f 值小的节点
    }
};


class Planner {
public:
    // 默认构造函数：初始化规划器
    Planner();

    // 定义AStar的启发式函数
    double heuristic_AStar(const Vector2i& current, const Vector2i& goal,
                       const unordered_set<Vector2i, Vector2iHash>& obstacles);

    // 定义启发式函数
    double heuristic(const State& current, const State& goal, 
                     const unordered_set<Vector2i, Vector2iHash>& obstacles_index);

    // 定义运动学约束
    State* Kinematic(const State& current, double delta, double step_size, double wheelbase, bool is_backward = false);

    // obstacles_index:障碍物索引
    // 广度优先搜索函数
    vector<Vector2i> Bfs(const Vector2i &start, const Vector2i &goal,
                                  const unordered_set<Vector2i, Vector2iHash>& obstacles);
    
    // A*启发式搜索函数
    vector<Vector2i> AStar(const Vector2i &start, const Vector2i &goal,
                                    const unordered_set<Vector2i, Vector2iHash>& obstacles);
    
    // Hybrid A*搜索函数
    // 给定运动学约束，使得搜索到的路径符合运动学模型
    // 扩展：使用A*搜索到的路径作为heuristic, 再使用Hybrid A*搜索
    std::vector<State*> HybridAStar(const State& start, const State& goal,
                                    const unordered_set<Vector2i, Vector2iHash>& obstacles_index,
                                    double wheelbase, double step_size, double max_steer);

    std::vector<State*> ReedsSheppPath(const State& start, const State& goal,
        const unordered_set<Vector2i, Vector2iHash>& obstacles,
        double wheelbase, double max_steer);
    
    bool CheckCollision(const State& state, 
        const unordered_set<Vector2i, Vector2iHash>& obstacles,
        double safety_radius);
};