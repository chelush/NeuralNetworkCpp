#pragma once

#include "core/math/Linalg.h"
#include <memory>

namespace nn {

enum class ActivationKind {
    ReLU,
    Sigmoid,
    Tanh,
};

class Activation {
public:
    explicit Activation(ActivationKind kind);

    MatrixXf forward(MatrixXf&& x);
    MatrixXf backward(MatrixXf&& grad_out);
    void zero_gradients();
    void clear_cache();

private:
    struct Cache {
        MatrixXf input;
        MatrixXf output;
    };

    ActivationKind kind_;
    std::unique_ptr<Cache> cache_;
};

}  // namespace nn
