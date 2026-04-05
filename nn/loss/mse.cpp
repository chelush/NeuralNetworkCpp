#include "nn/loss/mse.hpp"
#include <cassert>

namespace nn {

float MSE::value(const MatrixXf& pred, const MatrixXf& target) {
    assert(pred.rows() == target.rows() && pred.cols() == target.cols());
    const Eigen::Index n = pred.size();
    MatrixXf diff = pred - target;
    return diff.squaredNorm() / n;
}

MatrixXf MSE::backward(const MatrixXf& pred, const MatrixXf& target) {
    assert(pred.rows() == target.rows() && pred.cols() == target.cols());
    return 2.0f * (pred - target);
}

}  // namespace nn
