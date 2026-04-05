#include "nn/model/network_builder.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/sigmoid.hpp"
#include <cassert>

namespace nn {

NetworkBuilder NetworkBuilder::set_input(In n, Random& rnd) {
    return NetworkBuilder(n, rnd);
}

NetworkBuilder::NetworkBuilder(In n, Random& rnd) : rnd_(rnd), previous_(n.value), layers_() {
}

NetworkBuilder& NetworkBuilder::add_linear(Out m) {
    layers_.emplace_back(Linear(In{previous_}, m, rnd_, false));
    previous_ = m.value;
    return *this;
}

NetworkBuilder& NetworkBuilder::add_output_linear(Out m) {
    layers_.emplace_back(Linear(In{previous_}, m, rnd_, true));
    previous_ = m.value;
    return *this;
}

NetworkBuilder& NetworkBuilder::add_sigmoid() {
    assert(previous_ > 0);
    layers_.emplace_back(Sigmoid());
    return *this;
}

Network NetworkBuilder::extract() {
    assert(!layers_.empty() && "Network must contain at least one layer");
    return Network(std::move(layers_));
}

}  // namespace nn
