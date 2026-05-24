#pragma once

#include "core/math/Linalg.h"

namespace nn {

class BCE {
public:
    float value(const MatrixXf& pred, const MatrixXf& target);
    MatrixXf backward(const MatrixXf& pred, const MatrixXf& target);
};

}  // namespace nn
