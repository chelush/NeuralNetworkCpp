#pragma once

#include "core/math/Linalg.h"
#include "core/random/random.hpp"
#include "nn/layers/activation.hpp"
#include "nn/layers/layer.hpp"
#include "nn/model/network.hpp"
#include <vector>

namespace nn {

class NetworkBuilder {
public:
    static NetworkBuilder set_input(In n, Random& rnd);

    NetworkBuilder& add_linear(Out m);
    NetworkBuilder& add_activation(ActivationKind kind);
    NetworkBuilder& add_relu();
    NetworkBuilder& add_sigmoid();
    NetworkBuilder& add_tanh();

    Network extract();

private:
    explicit NetworkBuilder(In n, Random& rnd);

    Random& rnd_;
    Eigen::Index previous_;
    std::vector<Layer> layers_;
};

}  // namespace nn
