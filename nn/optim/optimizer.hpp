#pragma once

#include "nn/model/network.hpp"

namespace nn {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(Network& net, float learning_rate) = 0;
};

}  // namespace nn
