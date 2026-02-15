#pragma once

#include "core/math/types.hpp"
#include "nn/layers/layer.hpp"

namespace nn {

class Sigmoid : public Layer {
public:
  MatrixXf forward(const MatrixXf& x) override;
  MatrixXf backward(const MatrixXf& grad_out) override;

private:
  MatrixXf a_cache_;
};

}  // namespace nn
