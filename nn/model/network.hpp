#pragma once

#include "nn/layers/layer.hpp"
#include <vector>

namespace nn {

class NetworkBuilder;

class Network {
    friend class NetworkBuilder;

public:
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    Network(Network&&) noexcept;
    Network& operator=(Network&&) noexcept;

    MatrixXf forward(MatrixXf activations);
    MatrixXf backward(MatrixXf grad);
    void apply_gradients(float learning_rate);
    void zero_gradients();
    void clear_cache();

private:
    explicit Network(std::vector<Layer>&& layers);

    std::vector<Layer> layers_;
};

}  // namespace nn
