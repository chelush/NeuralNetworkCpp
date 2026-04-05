#pragma once

#include "core/math/Linalg.h"

namespace nn {

class MSE {
public:
    float value(const MatrixXf& pred, const MatrixXf& target);
    MatrixXf backward(const MatrixXf& pred, const MatrixXf& target);
};

}  // namespace nn
