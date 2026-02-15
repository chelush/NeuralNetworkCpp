#pragma once

#include "core/math/types.hpp"
#include <cassert>

namespace nn {

class MSE {
public:
  float value(const MatrixXf& pred, const MatrixXf& target);
  MatrixXf backward(const MatrixXf& pred, const MatrixXf& target);
};

}  // namespace nn
