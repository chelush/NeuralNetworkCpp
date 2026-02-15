#include "nn/loss/mse.hpp"
#include <Eigen/Dense>
#include <cassert>

namespace nn {

float MSE::value(const MatrixXf& pred, const MatrixXf& target) {
  assert(pred.rows() == target.rows() && pred.cols() == target.cols());
  MatrixXf diff = pred - target;
  return diff.squaredNorm() / static_cast<float>(pred.size());
}

MatrixXf MSE::backward(const MatrixXf& pred, const MatrixXf& target) {
  assert(pred.rows() == target.rows() && pred.cols() == target.cols());
  float n = static_cast<float>(pred.size());
  return 2.0f * (pred - target) / n;
}

}  // namespace nn
