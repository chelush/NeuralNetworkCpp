#pragma once

#include "nn/layers/linear.hpp"
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
    void zero_gradients();
    void clear_cache();
    const std::vector<Linear*>& linear_layers() const;

private:
    explicit Network(std::vector<Layer>&& layers);

    void rebuild_linear_cache_();

    std::vector<Layer> layers_;
    std::vector<Linear*> linear_cache_;
};

}  // namespace nn
