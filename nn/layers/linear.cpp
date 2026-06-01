#include "nn/layers/linear.hpp"
#include "app/except.h"

namespace nn {

Linear::Linear(In in_dim, Out out_dim, Random& rnd)
    : W_(rnd.uniform_matrix(out_dim, in_dim)), b_(rnd.uniform_vector(out_dim)) {
    NN_REQUIRE(in_dim.value > 0 && out_dim.value > 0, "Linear: dimensions must be positive");
}

MatrixXf Linear::forward(MatrixXf&& x) {
    NN_REQUIRE(x.cols() == W_.cols(), "Linear::forward(): input feature count mismatch");
    if (!cache_)
        cache_ = std::make_unique<Cache>();
    cache_->x = std::move(x);
    return (cache_->x * W_.transpose()).rowwise() + b_.transpose();
}

MatrixXf Linear::backward(MatrixXf&& grad_out) {
    NN_ASSERT(cache_ != nullptr, "Linear::backward(): forward() must be called first");
    NN_REQUIRE(grad_out.rows() == cache_->x.rows() && grad_out.cols() == W_.rows(),
               "Linear::backward(): gradient shape mismatch");
    cache_->dW = grad_out.transpose() * cache_->x;
    cache_->db = grad_out.colwise().sum().transpose();
    return std::move(grad_out) * W_;
}

void Linear::zero_gradients() {
    if (!cache_)
        return;
    cache_->dW.setZero();
    cache_->db.setZero();
}

void Linear::clear_cache() {
    cache_.reset();
}

void Linear::apply_parameter_step(const MatrixXf& delta_weights, const VectorXf& delta_bias) {
    NN_REQUIRE(delta_weights.rows() == W_.rows() && delta_weights.cols() == W_.cols(),
               "Linear::apply_parameter_step(): delta_weights shape mismatch");
    NN_REQUIRE(delta_bias.size() == b_.size(),
               "Linear::apply_parameter_step(): delta_bias shape mismatch");
    W_ += delta_weights;
    b_ += delta_bias;
}

const MatrixXf& Linear::weights() const {
    return W_;
}

const VectorXf& Linear::bias() const {
    return b_;
}

const MatrixXf& Linear::grad_weights() const {
    NN_ASSERT(cache_ != nullptr, "Linear::grad_weights(): backward() must be called first");
    return cache_->dW;
}

const VectorXf& Linear::grad_bias() const {
    NN_ASSERT(cache_ != nullptr, "Linear::grad_bias(): backward() must be called first");
    return cache_->db;
}

}  // namespace nn
