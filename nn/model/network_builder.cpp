#include "nn/model/network_builder.hpp"
#include "app/except.h"
#include "nn/layers/activation.hpp"
#include "nn/layers/linear.hpp"

namespace nn {

NetworkBuilder NetworkBuilder::set_input(In n, Random& rnd) {
    return NetworkBuilder(n, rnd);
}

NetworkBuilder::NetworkBuilder(In n, Random& rnd) : rnd_(rnd), previous_(n.value), layers_() {
    NN_REQUIRE(n.value > 0, "NetworkBuilder: input dimension must be > 0");
}

NetworkBuilder& NetworkBuilder::add_linear(Out m) {
    layers_.emplace_back(Linear(In{previous_}, m, rnd_));
    previous_ = m.value;
    return *this;
}

NetworkBuilder& NetworkBuilder::add_activation(ActivationKind kind) {
    NN_ASSERT(previous_ > 0, "NetworkBuilder::add_activation(): invalid previous layer width");
    layers_.emplace_back(Activation(kind));
    return *this;
}

NetworkBuilder& NetworkBuilder::add_relu() {
    return add_activation(ActivationKind::ReLU);
}

NetworkBuilder& NetworkBuilder::add_sigmoid() {
    return add_activation(ActivationKind::Sigmoid);
}

NetworkBuilder& NetworkBuilder::add_tanh() {
    return add_activation(ActivationKind::Tanh);
}

Network NetworkBuilder::extract() {
    NN_REQUIRE(!layers_.empty(), "NetworkBuilder::extract(): at least one layer is required");
    return Network(std::move(layers_));
}

}  // namespace nn
