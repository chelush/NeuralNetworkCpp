#pragma once

#include "core/random/random.hpp"
#include <memory>

namespace nn {

class Linear {
public:
    Linear(In in_dim, Out out_dim, Random& rnd);

    MatrixXf forward(MatrixXf&& x);
    MatrixXf backward(MatrixXf&& grad_out);
    void zero_gradients();
    void clear_cache();
    void apply_parameter_step(const MatrixXf& delta_weights, const VectorXf& delta_bias);

    const MatrixXf& weights() const;
    const VectorXf& bias() const;
    const MatrixXf& grad_weights() const;
    const VectorXf& grad_bias() const;

private:
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
