#include "nn/layers/sigmoid.hpp"
#include <cassert>

namespace nn {

MatrixXf Sigmoid::forward(MatrixXf&& x) {
    if (!cache_)
        cache_ = std::make_unique<Cache>();
    cache_->a.resize(x.rows(), x.cols());
    cache_->a = 1.0f / (1.0f + (-x.array()).exp());
    // Кэш нужен в backward; следующему слою — отдельная матрица (копия).
    return cache_->a;
}

MatrixXf Sigmoid::backward(MatrixXf&& grad_out) {
    assert(cache_ && "forward() must be called before backward()");
    return grad_out.array() * cache_->a.array() * (1.0f - cache_->a.array());
}

void Sigmoid::apply_gradients(float /*learning_rate*/) {
}

void Sigmoid::zero_gradients() {
}

void Sigmoid::clear_cache() {
    cache_.reset();
}

}  // namespace nn
