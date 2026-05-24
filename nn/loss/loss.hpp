#pragma once

#include "core/math/Linalg.h"

namespace nn {

class Loss {
public:
    virtual ~Loss() = default;

    virtual float value(const MatrixXf& pred, const MatrixXf& target) = 0;
    virtual MatrixXf backward(const MatrixXf& pred, const MatrixXf& target) = 0;
};

}  // namespace nn
