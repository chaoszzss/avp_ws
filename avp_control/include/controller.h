//
// Created by ubuntu on 25-3-8.
//

#ifndef AVP_WS_CONTROLLER_H
#define AVP_WS_CONTROLLER_H

#include <algorithm>
#include "KinematicModel.h"
#include <vector>
#include <cmath>


using namespace std;
#define PI 3.1415926

class Controller {
public:
    double calTargetIndex(const vector<double> &robot_state, const vector<vector<double>> &refer_path);

    double calTargetIndex(vector<double> robot_state, vector<vector<double>> refer_path, double l_d);

    double PIDController(const vector<double> &robot_state, const vector<vector<double>> &refer_path,
                                 const KinematicModel &ugv);

    double PurePursuitController(const vector<double> &robot_state, const vector<vector<double>> &refer_path,
                                                     const KinematicModel &ugv);

    double StanlyController(const vector<double> &robot_state, const vector<vector<double>> &refer_path,
                                                const KinematicModel &ugv);

    double LQRController(const vector<double> &robot_state, const vector<vector<double>> &refer_path,
                                             const KinematicModel &ugv);
};


#endif //AVP_WS_CONTROLLER_H
