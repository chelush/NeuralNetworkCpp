#include "nn/layers/linear.hpp"
#include <cassert>

namespace nn {

Linear::Linear(In in_dim, Out out_dim, Random& rnd, bool mean_loss_gradient_scaling)
    : mean_loss_gradient_scaling_(mean_loss_gradient_scaling),
      W_(rnd.uniform_matrix(out_dim, in_dim)),
      b_(rnd.uniform_vector(out_dim)) {
    assert(in_dim.value > 0 && out_dim.value > 0);
}

MatrixXf Linear::forward(MatrixXf&& x) {
    assert(x.cols() == W_.cols());
    if (!cache_)
        cache_ = std::make_unique<Cache>();
    cache_->x = std::move(x);
    return (cache_->x * W_.transpose()).rowwise() + b_.transpose();
}

MatrixXf Linear::backward(MatrixXf&& grad_out) {
    assert(cache_ && "forward() must be called before backward()");
    assert(grad_out.rows() == cache_->x.rows() && grad_out.cols() == W_.rows());
    if (mean_loss_gradient_scaling_)
        grad_out.array() *= (1.0f / grad_out.size());
    cache_->dW = grad_out.transpose() * cache_->x;
    cache_->db = grad_out.colwise().sum().transpose();
    return std::move(grad_out) * W_;
}

void Linear::apply_gradients(float learning_rate) {
    assert(cache_ && "backward() must be called before apply_gradients()");
    W_ -= learning_rate * cache_->dW;
    b_ -= learning_rate * cache_->db;
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

const MatrixXf& Linear::weights() const {
    return W_;
}
const VectorXf& Linear::bias() const {
    return b_;
}

}  // namespace nn
