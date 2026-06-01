#pragma once

#include "nn/loss/loss.hpp"

namespace nn {

class BCE final : public Loss {
public:
    float value(const MatrixXf& pred, const MatrixXf& target) override;
    MatrixXf backward(const MatrixXf& pred, const MatrixXf& target) override;
};

}  // namespace nn
