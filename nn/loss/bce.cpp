#include "nn/loss/bce.hpp"
#include "app/except.h"
#include <algorithm>
#include <cmath>

namespace nn {

float BCE::value(const MatrixXf& pred, const MatrixXf& target) {
    NN_REQUIRE(pred.rows() == target.rows() && pred.cols() == target.cols(),
               "BCE::value(): pred and target shapes must match");
    const float eps = 1e-7f;
    const Eigen::Index n = pred.size();

    float loss = 0.0f;
    for (Eigen::Index i = 0; i < pred.rows(); ++i) {
        for (Eigen::Index j = 0; j < pred.cols(); ++j) {
            const float p = std::clamp(pred(i, j), eps, 1.0f - eps);
            const float y = target(i, j);
            loss += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
        }
    }
    return loss / n;
}

MatrixXf BCE::backward(const MatrixXf& pred, const MatrixXf& target) {
    NN_REQUIRE(pred.rows() == target.rows() && pred.cols() == target.cols(),
               "BCE::backward(): pred and target shapes must match");
    const float eps = 1e-7f;
    const Eigen::Index n = pred.size();

    MatrixXf grad(pred.rows(), pred.cols());
    for (Eigen::Index i = 0; i < pred.rows(); ++i) {
        for (Eigen::Index j = 0; j < pred.cols(); ++j) {
            const float p = std::clamp(pred(i, j), eps, 1.0f - eps);
            const float y = target(i, j);
            grad(i, j) = (p - y) / (p * (1.0f - p) * n);
        }
    }
    return grad;
}

}  // namespace nn
