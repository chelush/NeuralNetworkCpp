#pragma once

#include "core/random/random.hpp"
#include <memory>

namespace nn {

class Linear {
public:
    Linear(In in_dim, Out out_dim, Random& rnd, bool mean_loss_gradient_scaling);

    MatrixXf forward(MatrixXf&& x);
    MatrixXf backward(MatrixXf&& grad_out);
    void apply_gradients(float learning_rate);
    void zero_gradients();
    void clear_cache();

    const MatrixXf& weights() const;
    const VectorXf& bias() const;

private:
    bool mean_loss_gradient_scaling_;
    MatrixXf W_;
    VectorXf b_;

    struct Cache {
        MatrixXf x;
        MatrixXf dW;
        VectorXf db;
    };
    std::unique_ptr<Cache> cache_;
};

}  // namespace nn
