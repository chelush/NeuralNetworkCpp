#include "nn/loss/mse.hpp"
#include "app/except.h"

namespace nn {

float MSE::value(const MatrixXf& pred, const MatrixXf& target) {
    NN_REQUIRE(pred.rows() == target.rows() && pred.cols() == target.cols(),
               "MSE::value(): pred and target shapes must match");
    NN_REQUIRE(pred.size() > 0, "MSE::value(): pred and target must be non-empty");
    const Eigen::Index n = pred.size();
    MatrixXf diff = pred - target;
    return diff.squaredNorm() / n;
}

MatrixXf MSE::backward(const MatrixXf& pred, const MatrixXf& target) {
    NN_REQUIRE(pred.rows() == target.rows() && pred.cols() == target.cols(),
               "MSE::backward(): pred and target shapes must match");
    NN_REQUIRE(pred.size() > 0, "MSE::backward(): pred and target must be non-empty");
    const Eigen::Index n = pred.size();
    return 2.0f * (pred - target) / n;
}

}  // namespace nn
