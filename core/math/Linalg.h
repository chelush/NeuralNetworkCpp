#pragma once

#include <Eigen/Dense>

namespace nn {

using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;

struct In {
    Eigen::Index value;
};
struct Out {
    Eigen::Index value;
};

}  // namespace nn
