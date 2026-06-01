#include "nn/optim/adam.hpp"
#include "app/except.h"

namespace nn {

Adam::Adam(float beta1, float beta2, float epsilon)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0), beta1_pow_(1.0f), beta2_pow_(1.0f),
      states_() {
    NN_REQUIRE(beta1_ >= 0.0f && beta1_ < 1.0f, "Adam: beta1 must be in [0, 1)");
    NN_REQUIRE(beta2_ >= 0.0f && beta2_ < 1.0f, "Adam: beta2 must be in [0, 1)");
    NN_REQUIRE(epsilon_ > 0.0f, "Adam: epsilon must be > 0");
}

void Adam::step(Network& net, float learning_rate) {
    NN_REQUIRE(learning_rate > 0.0f, "Adam::step(): learning_rate must be > 0");
    ++t_;
    beta1_pow_ *= beta1_;
    beta2_pow_ *= beta2_;
    std::size_t index = 0;
    const float bias_correction1 = 1.0f - beta1_pow_;
    const float bias_correction2 = 1.0f - beta2_pow_;

    for (Linear* linear_ptr : net.linear_layers()) {
        NN_ASSERT(linear_ptr != nullptr, "Adam::step(): null linear layer pointer");
        Linear& linear = *linear_ptr;
        if (states_.size() == index) {
            states_.push_back(State{MatrixXf::Zero(linear.weights().rows(), linear.weights().cols()),
                                    MatrixXf::Zero(linear.weights().rows(), linear.weights().cols()),
                                    VectorXf::Zero(linear.bias().size()),
                                    VectorXf::Zero(linear.bias().size())});
        }

        State& state = states_[index];
        NN_ASSERT(state.mW.rows() == linear.weights().rows() &&
                      state.mW.cols() == linear.weights().cols(),
                  "Adam::step(): state/weight shape mismatch");
        NN_ASSERT(state.mb.size() == linear.bias().size(),
                  "Adam::step(): state/bias shape mismatch");

        state.mW = beta1_ * state.mW + (1.0f - beta1_) * linear.grad_weights();
        state.vW =
            beta2_ * state.vW + (1.0f - beta2_) * linear.grad_weights().array().square().matrix();
        state.mb = beta1_ * state.mb + (1.0f - beta1_) * linear.grad_bias();
        state.vb =
            beta2_ * state.vb + (1.0f - beta2_) * linear.grad_bias().array().square().matrix();

        const MatrixXf mW_hat = state.mW / bias_correction1;
        const MatrixXf vW_hat = state.vW / bias_correction2;
        const VectorXf mb_hat = state.mb / bias_correction1;
        const VectorXf vb_hat = state.vb / bias_correction2;

        const MatrixXf delta_w =
            -learning_rate * mW_hat.array() / (vW_hat.array().sqrt() + epsilon_);
        const VectorXf delta_b =
            -learning_rate * mb_hat.array() / (vb_hat.array().sqrt() + epsilon_);
        linear.apply_parameter_step(delta_w, delta_b);

        ++index;
    }

    NN_ASSERT(index == states_.size(),
              "Adam::step(): optimizer state size mismatch for current network");
}

}  // namespace nn
