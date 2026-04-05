#pragma once

#include "nn/model/network.hpp"

namespace nn {

class SGD {
public:
    Network apply_gradient_step(float learning_rate, Network net);
};

}  // namespace nn
