#include "nn/model/network.hpp"

namespace nn {

void Network::add(Linear layer) {
  layers_.push_back(std::move(layer));
}

void Network::add(Sigmoid layer) {
  layers_.push_back(std::move(layer));
}

MatrixXf Network::forward(const MatrixXf& x) {
  MatrixXf out = x;
  for (auto& layer : layers_) {
    out = std::visit([&out](auto& l) { return l.forward(out); }, layer);
  }
  return out;
}

MatrixXf Network::backward(const MatrixXf& grad_out) {
  MatrixXf grad = grad_out;
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    grad = std::visit([&grad](auto& l) { return l.backward(grad); }, *it);
  }
  return grad;
}

void Network::for_each_linear(std::function<void(Linear&)> f) {
  for (auto& layer : layers_) {
    if (auto* p = std::get_if<Linear>(&layer))
      f(*p);
  }
}

}  // namespace nn
