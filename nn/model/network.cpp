#include "nn/model/network.hpp"
#include <cassert>

namespace nn {

void Network::add(std::unique_ptr<Layer> layer) {
  assert(layer != nullptr);
  layer_ptrs_.push_back(layer.get());
  layers_.push_back(std::move(layer));
}

MatrixXf Network::forward(const MatrixXf& x) {
  activations_.clear();
  activations_.push_back(x);
  MatrixXf out = x;
  for (Layer* l : layer_ptrs_) {
    out = l->forward(out);
    activations_.push_back(out);
  }
  return out;
}

MatrixXf Network::backward(const MatrixXf& grad_out) {
  MatrixXf grad = grad_out;
  for (int i = static_cast<int>(layer_ptrs_.size()) - 1; i >= 0; --i)
    grad = layer_ptrs_[i]->backward(grad);
  return grad;
}

}  // namespace nn
