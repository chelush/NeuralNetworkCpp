#pragma once

#include "core/math/types.hpp"

namespace nn {

class Layer {
public:
  virtual ~Layer() = default;
  virtual MatrixXf forward(const MatrixXf& x) = 0;
  virtual MatrixXf backward(const MatrixXf& grad_out) = 0;
};

}  // namespace nn
