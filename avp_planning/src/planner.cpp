//
// Created by ubuntu on 25-3-2.
//
#include "../include/planner.h"

double resolution = 0.5;  // 地图分辨率

Planner::Planner() {

}


vector<Vector2i> Planner::Bfs(const Vector2i &start, const Vector2i &goal,
                     const unordered_set<Vector2i, Vector2iHash>& obstacles) {
        
    // 处理特殊输入情况
    if (start == goal) {return vector<Vector2i>{start};}  // 起点即终点
    // 在障碍物集合中搜索起点坐标，找到指向元素，没找到指向end，不等于end表示找到
    if (obstacles.find(start) != obstacles.end()) return {};  // 起点在障碍物中
    // 在障碍物集合中搜索终点坐标，找到指向元素，没找到指向end，不等于end表示找到
    if (obstacles.find(goal) != obstacles.end()) return {};  // 终点在障碍物中

    // 队列指针初始化
    std::queue<State*> open_table;
    // 插入起点
    open_table.push(new State(start.x(), start.y(), 0, 0, 0, nullptr, false));

    // 已访问集合初始化
    std::unordered_set<Vector2i, Vector2iHash> close_table;
    // 将起点加入到已访问集合
    close_table.insert(start);

    // 前驱映射初始化（用于回溯路径）
    // 键值对：<邻居坐标，前驱节点指针>
    // 前驱映射通常用来记录每个节点的父节点，这样在找到目标后可以回溯路径
    std::unordered_map<Vector2i, State*, Vector2iHash> parent_map;
    parent_map[start] = nullptr; // 起点没有前驱节点


    // 循环遍历队列
    while (!open_table.empty()) {
        // 取出列前端节点
        State* current = open_table.front();
        // 弹出队列前端节点
        open_table.pop();

        // 如果current是目标，则跳出循环
        if (current->x == goal.x() && current->y == goal.y()) {
            break;
        }
        // 否则，遍历当前节点四个方向生成邻居坐标
        int dx[] = {-1, 1, 0, 0, -1, 1, -1, 1};
        int dy[] = {0, 0, -1, 1, -1, -1, 1, 1};
        for (int i = 0; i < 8; i++) {
            // 邻居坐标
            // Vector2i neighbor = Vector2i(current->x + dx[i], current->y + dy[i]);
            // 邻居指针
            State* neighbor = new State(current->x + static_cast<double>(dx[i]), 
                                        current->y + static_cast<double>(dy[i]), 0, 0, 0, current, false);
            // 检查是否在障碍物中
            if (obstacles.find(Vector2i(neighbor->x, neighbor->y)) != obstacles.end()) continue;
            // 检查是否在已访问合集中
            if (close_table.find(Vector2i(neighbor->x, neighbor->y)) != close_table.end()) continue;
            // 检查是否越界?

            // 标记邻居为已访问（需要提前标记，否则可能重复入队）
            close_table.insert(Vector2i(neighbor->x, neighbor->y));
            // 添加到队列（一旦入队，这个节点一定会被处理，所以需要提前标记为已访问）
            open_table.push(neighbor);
            // 更新邻居的前驱
            parent_map[Vector2i(neighbor->x, neighbor->y)] = current;
        }
    }

    // 如果目标节点没有被访问到，返回空路径
    if (parent_map.find(goal) == parent_map.end()) return {};

    // 回溯路径
    std::vector<Vector2i> path;
    State* current_state = parent_map[goal];
    path.push_back(goal); // 添加终点
    while (current_state != nullptr) {
        path.push_back(Vector2i(current_state->x, current_state->y));
        current_state = current_state->parent;
    }

    // 反转路径
    std::reverse(path.begin(), path.end());

    // 释放内存-->不需要手动释放内存，parent_map是智能指针，会自动管理内存的申请与释放(x)
    // for (auto& entry : parent_map) {
    //     delete entry.second;
    // }

    return path;
}

/**
 * BFS和A*中关于parent_map的区别
 * 
 * BFS:
 * 单纯记录节点的发现路径
 * 每个坐标首次被访问时记录父节点
 * 后续不再更新（因 BFS 保证首次访问即最短路径）
 * BFS[BFS parent_map特性] -->|单次记录| B1[每个坐标只记录一次]
 * BFS -->|无代价比较| B2[父节点不可变]
 * BFS -->|内存管理简单| B3[无需动态释放]
 * 
 * A*:
 * 维护每个坐标的最优路径 
 * 持续比较不同路径的代价（g值）
 * 支持覆盖旧记录（当发现更优路径时）
 * ASTAR[A* parent_map特性] -->|动态更新| A1[允许覆盖旧记录]
 * ASTAR -->|代价敏感| A2[持续比较g值]
 * ASTAR -->|内存复杂| A3[需主动释放旧节点]
 * 
 */

double Planner::heuristic_AStar(const Vector2i& current, const Vector2i& goal,
                               const unordered_set<Vector2i, Vector2iHash>& obstacles) {
    
    double dx = current.x() - goal.x();
    double dy = current.y() - goal.y();

    // 权重参数
    const double k_pos = 1.0;
    const double k_distance = 0.5;

    // 安全区域参数重定义
    const int safety_cells = 3;          // 根据分辨率调整安全距离
    const double obstacle_penalty_value = 200.0; // 调整惩罚值
    double obstacle_penalty = 0.0;


    // 计算当前节点对应的网格单元
    Vector2i current_cell(
        static_cast<int>(current.x() / resolution),
        static_cast<int>(current.y() / resolution)
    );

    // 检查当前节点周围的障碍物（重要修正：循环范围改为 <=）
    bool is_near_obstacle = false;
    for (int dx = -safety_cells; dx <= safety_cells; ++dx) {
        for (int dy = -safety_cells; dy <= safety_cells; ++dy) {
            Vector2i neighbor_cell(current_cell.x() + dx, current_cell.y() + dy);
            if (obstacles.find(neighbor_cell) != obstacles.end()) {
                is_near_obstacle = true;
                break;
            }
        }
        if (is_near_obstacle) break;
    }

    // 如果在安全距离内，添加惩罚项
    if (is_near_obstacle) {
        obstacle_penalty = obstacle_penalty_value;
    }

    return k_pos * std::hypot(dx, dy) + k_distance * obstacle_penalty;
}



vector<Vector2i> Planner::AStar(const Vector2i &start, const Vector2i &goal,
                                const unordered_set<Vector2i, Vector2iHash>& obstacles) {
    
    // 处理特殊情况
    if (start == goal) {return vector<Vector2i>{start};}      // 起点是终点
    if (obstacles.find(start) != obstacles.end()) return {};  // 起点在障碍物中
    if (obstacles.find(goal) != obstacles.end()) return {};   // 终点在障碍物中

    // **初始化优先队列, 按照f值排序
    std::priority_queue<State*, std::vector<State*>, CompareState> open_table;
    // 计算起点的启发式代价
    double h_start = std::hypot(start.x() - goal.x(), start.y() - goal.y());
    // 将起点添加进队列
    State* start_state = new State(start.x(), start.y(), 0, 0, h_start, nullptr, false);
    open_table.push(start_state);

    // 初始化已经访问的合集
    std::unordered_set<Vector2i, Vector2iHash> close_table;
    // 将起点添加到已经访问的合集中
    // close_table.insert(start);

    // 定义前驱映射<Vector2i, *State>类型的<key, Value>对
    std::unordered_map<Vector2i, State*, Vector2iHash> parent_map;
    // 设置起点的前驱节点（由于起点的h和f不为0，所以需要设置）
    parent_map[start] = start_state;

    // 定义路径
    vector<Vector2i> path;

    // 循环遍历队列
    while (!open_table.empty()) {
        // 取出队列前端节点
        State* current = open_table.top();
        // 弹出前端节点
        open_table.pop();

        // 判断是否已经访问过
        if (close_table.find(Vector2i(current->x, current->y)) != close_table.end()) {
            // 注意：不要手动释放内存-->容易出现双重释放内存而报错的现象
            continue;
        }
        // 将当前节点加入已访问合集
        close_table.insert(Vector2i(current->x, current->y));

        // 如果当前节点是目标节点, 回溯路径
        if (current->x == goal.x() && current->y == goal.y()) {
            State* state = current;
            // path.push_back(goal);
            while (state != nullptr) {
                path.push_back(Vector2i(state->x, state->y));
                state = state->parent;
            }
            // 反转路径
            std::reverse(path.begin(), path.end());
            // 释放内存
            break;
        }

        // 否则，遍历当前节点四个方向生成邻居坐标
        int dx[] = {-1, 1, 0, 0, -1, 1, -1, 1};
        int dy[] = {0, 0, -1, 1, -1, -1, 1, 1};
        for (int i = 0; i < 8; i++) {
            // 设置邻居坐标
            Vector2i neighbor = Vector2i(current->x + dx[i], current->y + dy[i]);

            // 增加安全距离检测


            // 检查是否在障碍物中
            if (obstacles.find(neighbor) != obstacles.end()) continue;
            // 检查是否在已经访问的合集中
            if (close_table.find(neighbor) != close_table.end()) continue;

            // 计算邻居节点代价
            double g_neighbor = current->g + (i < 4 ? 1.0 : std::sqrt(2)); // 对角线方向代价更高
            // double g_neighbor = current->g + 1.0;
            // 曼哈顿距离
            // double h_neighbor = heuristic_AStar(neighbor, goal, obstacles);
            double h_neighbor = std::hypot(neighbor.x() - goal.x(), neighbor.y() - goal.y());
            // 欧几里得距离
            // double h_neighbor = std::hypot(neighbor.x() - goal.x(), neighbor.y() - goal.y());
            // double f_neighbor = g_neighbor + h_neighbor;

            // 检查邻居节点是否在队列中
            auto it = parent_map.find(neighbor); // 迭代器
            // 如果邻居节点在队列中，并且g值大于当前g值，则跳过
            // it->second->g：之前记录的到达该邻居坐标的历史最小累计代价
            // 当前路径到达该邻居的新累计代价（current->g + 1）
            if (it != parent_map.end() && it->second->g > g_neighbor) {
                // delete it->second;  //释放旧节点
                parent_map.erase(it);
            }

            // current:当前节点
            // it->second:之前路径到达该邻居的节点坐标
            // neighbor_state:当前路径达到该邻居的节点坐标
            // 这两个节点的坐标值是相同的，但是g不同，需要更新该节点的前驱

            // 否则创建新的节点
            // 这个新的节点相较于it->second来说，变化的地方有累积代代价g和父节点parent（也就是current）
            State* neighbor_state = new State(neighbor.x(), neighbor.y(), 0, g_neighbor, h_neighbor, current, false);
            // 更新邻居节点的坐标->节点的映射关系，不是前驱关系
            // 真正的父节点信息存储在State对象的parent成员中
            // 这样设计的原因是：涉及不同路径的切换时，只需要更新parent信息，而不需要更新坐标
            parent_map[neighbor] = neighbor_state;
            // 必须将新节点加入到优先队列中
            open_table.push(neighbor_state);
        }
    }

    // 修复：增加队列清理
    // while (!open_table.empty()) {
    //     delete open_table.top();
    //     open_table.pop();
    // }

    return path;
}

// 结合位置和角度的启发式函数
double Planner::heuristic(const State& current, const State& goal, 
                          const unordered_set<Vector2i, Vector2iHash>& obstacles_index) {
    double dx = current.x - goal.x;
    double dy = current.y - goal.y;
    double dtheta = std::abs(current.theta - goal.theta); 

    // 权重参数
    const double k_pos = 1.0;
    const double k_theta = 0.8;
    const double k_smoothness = 1.0; // 新增平滑度权重
    const double k_distance = 0.5; // 新增到障碍物的距离权重

    // 添加朝向一致性奖励
    double heading_diff = std::abs(std::atan2(dy, dx) - current.theta);
    double heading_bonus = std::cos(heading_diff); // 方向一致时接近1

    // 计算路径平滑度惩罚项
    double smoothness_penalty = 0.0;
    if (current.parent != nullptr) {
        double prev_dtheta = std::abs(current.parent->theta - current.theta);
        smoothness_penalty = k_smoothness * prev_dtheta;
    }

    // 障碍物惩罚新增:
    // 安全距离参数
    const int safety_radius_cells = 2; // 根据地图分辨率调整
    const double obstacle_penalty_value = 200.0; // 惩罚值
    double obstacle_penalty = 0.0;
    double resolution = 0.5;  // 地图分辨率

    Vector2i current_cell(
        static_cast<int>(current.x / resolution),
        static_cast<int>(current.y / resolution));

    // 检查当前状态周围的 blocks 是否存在障碍物
    // 在以当前节点为中心的safety_radius_cells正方形区域内搜索
    bool is_near_obstacle = false;
    for (int dx = -safety_radius_cells; dx <= safety_radius_cells; ++dx) {
        for (int dy = -safety_radius_cells; dy <= safety_radius_cells; ++dy) {
            Vector2i neighbor_cell(current_cell.x() + dx, current_cell.y() + dy);
            if (obstacles_index.find(neighbor_cell) != obstacles_index.end()) {
                is_near_obstacle = true;
                break;
            }
        }
        if (is_near_obstacle) break;
    }

    if (is_near_obstacle) {
        obstacle_penalty = obstacle_penalty_value; // 加入惩罚
    }

    // 计算总启发式代价值
    return k_pos * std::hypot(dx, dy) 
        + k_theta * dtheta 
        - heading_bonus 
        + smoothness_penalty 
        + k_distance * obstacle_penalty;
}


/**
 * 运动学模型（状态预测表达式）
 * 参数说明
 * delta：转向角：前轮与车辆纵轴的夹角
 * step_size：步长：离散时间步长内车辆前进的距离，step_size=v*dt
 * wheelbase：轮距:前轮到后轮的距离
 */
State* Planner::Kinematic(const State& current, double delta, double step_size, double wheelbase, bool is_backward) {
    // 计算新的状态坐标和角度
    // double beta = std::atan(wheelbase * std::tan(delta) / step_size); // 侧偏角
    double dx = step_size * std::cos(current.theta + delta / 2);   // x坐标增量
    double dy = step_size * std::sin(current.theta + delta / 2);   // y坐标增量
    double dtheta = (step_size / wheelbase) * std::tan(delta);  // 角度增量

    // 注意：当车辆转弯时，由于轮距的存在，车辆并不会完全沿着前轮的方向行驶，
    // 而是有一个侧偏角beta。这个角度是计算出来的，而不是直接控制的，它由delta、轮距和前进距离共同决定

    // 更新状态
    // theta：朝向角（车辆的朝向，车辆当前的前进方向）
    // beta：侧偏角（车辆的实际行进方向与车辆纵轴之间的夹角，由delta、轮距和前进距离共同决定）
    // delta：转向角（前轮相对于车辆纵轴的偏转角度，由方向盘控制，决定了车辆的转向方向和程度）
    return new State(
        current.x + dx,
        current.y + dy,
        current.theta + dtheta,
        current.g,
        0,
        const_cast<State*>(&current),
        current.is_backward // 运动方向标识
    );
}

// 检查是否在障碍物附近
bool isNearObstacle(const Vector2i& pos, const unordered_set<Vector2i, Vector2iHash>& obstacles_index, double safety_distance) {
    for (const auto& obs : obstacles_index) {
        if (std::hypot(pos.x() - obs.x(), pos.y() - obs.y()) < safety_distance) {
            return true;
        }
    }
    return false;
}


/**
 * Hybrid A*算法：相较于A*算法，变动的地方是：1、寻找邻居节点时加入运动学约束 2、
 * 参数说明
 * wheelbase:轮距
 * step_size:步长
 * max_steer：最大转向角(delta角度的范围)
 */
std::vector<State*> Planner::HybridAStar(const State& start, const State& goal,
                                const unordered_set<Vector2i, Vector2iHash>& obstacles_index,
                                double wheelbase, double step_size, double max_steer) {
    
    // 处理特殊情况
    // {const_cast<State*>(&start)}：将start转换为State*类型
    if (start.x == goal.x && start.y == goal.y) return {const_cast<State*>(&start)}; 
    if (obstacles_index.find(Vector2i(start.x, start.y)) != obstacles_index.end()) return {};
    if (obstacles_index.find(Vector2i(goal.x, goal.y)) != obstacles_index.end()) return {};

    // 添加超时机制
    int max_iterations = 50000; // MODIFIED

    // 初始化优先队列
    std::priority_queue<State*, std::vector<State*>, CompareState> open_table;
    // 初始化起点
    State* start_state = new State(start.x, start.y, start.theta, 0, heuristic(start, goal, obstacles_index), nullptr,
                                    false);     // 新增运动方向标识
    // 将起点加入优先队列
    open_table.push(start_state);

    // 初始化已访问合集(健类型修改为StateKey)
    std::unordered_set<StateKey, StateKeyHash> close_table;

    // 初始化前驱映射(用于查找邻居节点是否存在于队列中)
    std::unordered_map<StateKey, State*, StateKeyHash> parent_map;
    parent_map[StateKey(Vector2i(start.x, start.y), start.theta)] = start_state;

    // 定义路径
    std::vector<State*> path;

    // 循环遍历队列
    while (!open_table.empty() && max_iterations-- > 0) {
        // 取出队列前端节点
        State* current = open_table.top();
        // 弹出前端节点
        open_table.pop();

        // 使用量化后的状态检查
        StateKey current_key(Vector2i(current->x, current->y), current->theta);
        // 判断是否已经访问过
        if (close_table.find(current_key) != close_table.end()) {
            continue;
        }
        // 将当前节点加入已访问合集
        close_table.insert(current_key);

        // 如果当前节点是目标节点，回溯路径
        // 完善终止条件（位置+角度容差）
        if (std::hypot(current->x - goal.x, current->y - goal.y) < 1.0 && 
        std::abs(current->theta - goal.theta) < M_PI ) {
            State* state = current;
            while (state) {
                path.push_back(state);
                state = state->parent;
            }
            // 反转路径
            std::reverse(path.begin(), path.end());
            // 释放内存
            break;
        }

        // 否则，生成邻居节点(邻居节点的生成需要考虑运动学约束)
        // 生成邻居节点的转向角度
        // for (double step : {step_size, -step_size * 0.5}) {
        for (double delta = -max_steer; delta <= max_steer; delta += max_steer / 4) {
        // 使用运动学模型生成邻居节点
        State* neighbor = Kinematic(*current, delta, step_size, wheelbase, current->is_backward);
        
        // 检查是否在障碍物中
        double resolution = 0.5;  // 假设分辨率固定为0.5米（需确保与avp_planning_node中的值一致）
        // 量化邻居节点坐标
        Vector2i neighbor_pos(
            static_cast<int>(round(neighbor->x / resolution)),
            static_cast<int>(round(neighbor->y / resolution))
        );
        if (obstacles_index.find(neighbor_pos) != obstacles_index.end()) continue;

        // 检查邻居节点和障碍物之间的安全距离
        // 这个逻辑太过简单粗暴，需要进一步优化
        // if (isNearObstacle(neighbor_pos, obstacles_index, 2.0)) continue;

        // 检查量化状态是否在已访问合集中 / 
        StateKey neighbor_key(Vector2i(neighbor->x, neighbor->y), neighbor->theta);
        if (close_table.find(neighbor_key) != close_table.end()) continue;

        // 计算邻居节点的代价
        neighbor->g = current->g + step_size; // MODIFIED
        neighbor->h = heuristic(*neighbor, goal, obstacles_index);
        neighbor->f = neighbor->g + neighbor->h;
        neighbor->parent = current;

        // 第1步：查找该节点是否已经在 parent_map 中
        auto it = parent_map.find(neighbor_key);
        // 更新代价最小的路径
        if (it != parent_map.end() && it->second->g <= neighbor->g) {
            // delete neighbor;  //释放旧节点
            continue;
        }

        // 更新邻居节点的前驱
        parent_map[neighbor_key] = neighbor;
        // 将邻居节点加入到优先队列中
        open_table.push(neighbor); 
        }
    }

    // 清理内存
    while (!open_table.empty()) {
        delete open_table.top();
        open_table.pop();
    }

    // 返回路径
    return path;
}


/**
 * 基于RS曲线的路径规划算法
 */
// 在Planner类中添加RS曲线路径规划方法
// std::vector<State*> Planner::ReedsSheppPath(const State& start, const State& goal,
//                                            const unordered_set<Vector2i, Vector2iHash>& obstacles_index,
//                                            double wheelbase, double max_curvature) {
//     // 参数校验
//     if (obstacles_index.find(Vector2i(start.x, start.y)) != obstacles_index.end() ||
//         obstacles_index.find(Vector2i(goal.x, goal.y)) != obstacles_index.end()) {
//         return {};
//     }

//     // 生成RS路径
//     ReedsSheppStateSpace::ReedsSheppPath path;
//     ReedsSheppStateSpace stateSpace(max_curvature);
//     double q0[3] = {start.x, start.y, start.theta};
//     double q1[3] = {goal.x, goal.y, goal.theta};
//     stateSpace.sample(q0, q1, 0.1, path); // 0.1为步长

//     // 将路径转换为State序列
//     std::vector<State*> rs_path;
//     State* prev_state = new State(start.x, start.y, start.theta, 0, 0, nullptr, false);
//     for (const auto& seg : path) {
//         for (double t = 0; t <= seg.length; t += 0.1) {
//             double x = prev_state->x + t * std::cos(prev_state->theta);
//             double y = prev_state->y + t * std::sin(prev_state->theta);
//             double theta = prev_state->theta + t * seg.steer / wheelbase;
            
//             // 检查碰撞
//             Vector2i grid_pos(static_cast<int>(x/resolution), 
//                       static_cast<int>(y/resolution));
//             if (obstacles_index.find(grid_pos) != obstacles_index.end()) {
//                 // 路径被障碍物阻挡，返回空路径
//                 for (auto s : rs_path) delete s;
//                 return {};
//             }
            
//             State* new_state = new State(x, y, theta, 0, 0, prev_state, false);
//             rs_path.push_back(new_state);
//             prev_state = new_state;
//         }
//     }

//     // 添加终点状态
//     rs_path.push_back(new State(goal.x, goal.y, goal.theta, 0, 0, prev_state, false));
//     return rs_path;
// }

// // 新增碰撞检测辅助函数（需在头文件中声明）
// bool Planner::CheckCollision(const State& state, 
//                             const unordered_set<Vector2i, Vector2iHash>& obstacles,
//                             double safety_radius) {
//     Vector2i grid_pos(
//         static_cast<int>(state.x / resolution),
//         static_cast<int>(state.y / resolution)
//     );
    
//     // 扩展检测区域
//     for (int dx = -1; dx <= 1; ++dx) {
//         for (int dy = -1; dy <= 1; ++dy) {
//             Vector2i check_pos(grid_pos.x() + dx, grid_pos.y() + dy);
//             if (obstacles.find(check_pos) != obstacles.end()) {
//                 return true;
//             }
//         }
//     }
//     return false;
// }