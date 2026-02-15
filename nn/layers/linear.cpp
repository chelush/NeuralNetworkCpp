#include "nn/layers/linear.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <random>

namespace nn {

Linear::Linear(In in_dim, Out out_dim)
    : in_features_(in_dim.value), out_features_(out_dim.value),
      W_(static_cast<Eigen::Index>(out_dim.value), static_cast<Eigen::Index>(in_dim.value)),
      b_(static_cast<Eigen::Index>(out_dim.value)),
      dW_(static_cast<Eigen::Index>(out_dim.value), static_cast<Eigen::Index>(in_dim.value)),
      db_(static_cast<Eigen::Index>(out_dim.value)) {
  assert(in_features_ > 0 && out_features_ > 0);
  std::default_random_engine rng(42);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  for (int i = 0; i < W_.size(); ++i)
    W_.data()[i] = dist(rng);
  b_.setZero();
}

MatrixXf Linear::forward(const MatrixXf& x) {
  assert(static_cast<std::size_t>(x.cols()) == in_features_);
  x_cache_ = x;
  return x * W_.transpose() + b_.transpose().replicate(x.rows(), 1);
}

MatrixXf Linear::backward(const MatrixXf& grad_out) {
  assert(grad_out.rows() == x_cache_.rows() && static_cast<std::size_t>(grad_out.cols()) == out_features_);
  dW_ = grad_out.transpose() * x_cache_;
  db_ = grad_out.colwise().sum().transpose();
  return grad_out * W_;
}

}  // namespace nn
