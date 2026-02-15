#pragma once

#include "core/math/types.hpp"
#include "nn/layers/layer.hpp"

namespace nn {

class Linear : public Layer {
public:
  Linear(In in_dim, Out out_dim);
  MatrixXf forward(const MatrixXf& x) override;
  MatrixXf backward(const MatrixXf& grad_out) override;

  const MatrixXf& weights() const { return W_; }
  const VectorXf& bias() const { return b_; }
  MatrixXf& weights() { return W_; }
  VectorXf& bias() { return b_; }
  const MatrixXf& grad_weights() const { return dW_; }
  const VectorXf& grad_bias() const { return db_; }

private:
  std::size_t in_features_;
  std::size_t out_features_;
  MatrixXf W_;
  VectorXf b_;
  MatrixXf dW_;
  VectorXf db_;
  MatrixXf x_cache_;
};

}  // namespace nn
