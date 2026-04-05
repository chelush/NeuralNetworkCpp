#pragma once

#include "core/math/Linalg.h"
#include <memory>

namespace nn {

class Sigmoid {
public:
    MatrixXf forward(MatrixXf&& x);
    MatrixXf backward(MatrixXf&& grad_out);
    void apply_gradients(float learning_rate);
    void zero_gradients();
    void clear_cache();

private:
    struct Cache {
        MatrixXf a;
    };
    std::unique_ptr<Cache> cache_;
};

}  // namespace nn
