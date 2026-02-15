#include "nn/layers/sigmoid.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace nn {

MatrixXf Sigmoid::forward(const MatrixXf& x) {
  a_cache_ = 1.0f / (1.0f + (-x.array()).exp());
  return a_cache_;
}

MatrixXf Sigmoid::backward(const MatrixXf& grad_out) {
  return grad_out.array() * a_cache_.array() * (1.0f - a_cache_.array());
}

}  // namespace nn
