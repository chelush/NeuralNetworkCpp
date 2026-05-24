#include "nn/model/network.hpp"
#include "app/except.h"

namespace nn {

Network::Network(std::vector<Layer>&& layers) : layers_(std::move(layers)) {
    NN_REQUIRE(!layers_.empty(), "Network: must contain at least one layer");
}

Network::Network(Network&&) noexcept = default;

Network& Network::operator=(Network&&) noexcept = default;

MatrixXf Network::forward(MatrixXf activations) {
    NN_ASSERT(!layers_.empty(), "Network::forward(): network has no layers");
    for (auto& layer : layers_)
        activations = layer.forward(std::move(activations));
    return activations;
}

MatrixXf Network::backward(MatrixXf grad) {
    NN_ASSERT(!layers_.empty(), "Network::backward(): network has no layers");
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        grad = it->backward(std::move(grad));
    return grad;
}

void Network::zero_gradients() {
    for (auto& layer : layers_)
        layer.zero_gradients();
}

void Network::clear_cache() {
    for (auto& layer : layers_)
        layer.clear_cache();
}

std::vector<Linear*> Network::linear_layers() {
    std::vector<Linear*> linear;
    linear.reserve(layers_.size());
    for (auto& layer : layers_) {
        if (Linear* ptr = layer.as_linear(); ptr != nullptr)
            linear.push_back(ptr);
    }
    return linear;
}

std::vector<const Linear*> Network::linear_layers() const {
    std::vector<const Linear*> linear;
    linear.reserve(layers_.size());
    for (const auto& layer : layers_) {
        if (const Linear* ptr = layer.as_linear(); ptr != nullptr)
            linear.push_back(ptr);
    }
    return linear;
}

}  // namespace nn
