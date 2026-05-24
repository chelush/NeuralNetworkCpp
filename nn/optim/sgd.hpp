#pragma once

#include "nn/optim/optimizer.hpp"

namespace nn {

class SGD final : public Optimizer {
public:
    void step(Network& net, float learning_rate) override;
};

}  // namespace nn
