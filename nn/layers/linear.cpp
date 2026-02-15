#include "nn/layers/linear.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <random>

namespace nn {

Linear::Linear(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features),
      W_(out_features_, in_features_), b_(out_features_),
      dW_(out_features_, in_features_), db_(out_features_) {
  assert(in_features_ > 0 && out_features_ > 0);
  std::default_random_engine rng(42);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  for (int i = 0; i < W_.size(); ++i)
    W_.data()[i] = dist(rng);
  b_.setZero();
}

MatrixXf Linear::forward(const MatrixXf& x) {
  assert(x.cols() == in_features_);
  x_cache_ = x;
  return x * W_.transpose() + b_.transpose().replicate(x.rows(), 1);
}

MatrixXf Linear::backward(const MatrixXf& grad_out) {
  assert(grad_out.rows() == x_cache_.rows() && grad_out.cols() == out_features_);
  dW_ = grad_out.transpose() * x_cache_;
  db_ = grad_out.colwise().sum().transpose();
  return grad_out * W_;
}

}  // namespace nn
