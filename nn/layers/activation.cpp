#include "nn/layers/activation.hpp"
#include "app/except.h"
#include <stdexcept>

namespace nn {

Activation::Activation(ActivationKind kind) : kind_(kind), cache_() {
}

MatrixXf Activation::forward(MatrixXf&& x) {
    if (!cache_)
        cache_ = std::make_unique<Cache>();

    cache_->input = std::move(x);
    switch (kind_) {
        case ActivationKind::ReLU:
            cache_->output = cache_->input.cwiseMax(0.0f);
            break;
        case ActivationKind::Sigmoid:
            cache_->output = 1.0f / (1.0f + (-cache_->input.array()).exp());
            break;
        case ActivationKind::Tanh:
            cache_->output = cache_->input.array().tanh();
            break;
    }
    return cache_->output;
}

MatrixXf Activation::backward(MatrixXf&& grad_out) {
    NN_ASSERT(cache_ != nullptr, "Activation::backward(): forward() must be called first");
    NN_REQUIRE(grad_out.rows() == cache_->output.rows() && grad_out.cols() == cache_->output.cols(),
               "Activation::backward(): gradient shape mismatch");

    switch (kind_) {
        case ActivationKind::ReLU:
            return grad_out.array() * (cache_->input.array() > 0.0f).cast<float>();
        case ActivationKind::Sigmoid:
            return grad_out.array() * cache_->output.array() * (1.0f - cache_->output.array());
        case ActivationKind::Tanh:
            return grad_out.array() * (1.0f - cache_->output.array().square());
    }

    throw std::logic_error("unknown activation kind");
}

void Activation::zero_gradients() {
}

void Activation::clear_cache() {
    cache_.reset();
}

}  // namespace nn
