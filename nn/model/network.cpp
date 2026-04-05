#include "nn/model/network.hpp"
#include <cassert>

namespace nn {

Network::Network(std::vector<Layer>&& layers) : layers_(std::move(layers)) {
    assert(!layers_.empty() && "Network must be built with at least one layer");
}

Network::Network(Network&&) noexcept = default;

Network& Network::operator=(Network&&) noexcept = default;

MatrixXf Network::forward(MatrixXf activations) {
    assert(!layers_.empty());
    for (auto& layer : layers_)
        activations = layer.forward(std::move(activations));
    return activations;
}

MatrixXf Network::backward(MatrixXf grad) {
    assert(!layers_.empty());
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        grad = it->backward(std::move(grad));
    return grad;
}

void Network::apply_gradients(float learning_rate) {
    for (auto& layer : layers_)
        layer.apply_gradients(learning_rate);
}

void Network::zero_gradients() {
    for (auto& layer : layers_)
        layer.zero_gradients();
}

void Network::clear_cache() {
    for (auto& layer : layers_)
        layer.clear_cache();
}

}  // namespace nn
