#pragma once

#include "core/math/types.hpp"
#include "nn/layers/layer.hpp"

namespace nn {

class Linear : public Layer {
public:
  Linear(int in_features, int out_features);
  MatrixXf forward(const MatrixXf& x) override;
  MatrixXf backward(const MatrixXf& grad_out) override;

  const MatrixXf& weights() const { return W_; }
  const VectorXf& bias() const { return b_; }
  MatrixXf& weights() { return W_; }
  VectorXf& bias() { return b_; }
  const MatrixXf& grad_weights() const { return dW_; }
  const VectorXf& grad_bias() const { return db_; }

private:
  int in_features_;
  int out_features_;
  MatrixXf W_;
  VectorXf b_;
  MatrixXf dW_;
  VectorXf db_;
  MatrixXf x_cache_;
};

}  // namespace nn
