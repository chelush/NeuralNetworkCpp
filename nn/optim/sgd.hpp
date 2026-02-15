#pragma once

#include "nn/model/network.hpp"

namespace nn {

class SGD {
public:
  void step(Network& net, float lr);
};

}  // namespace nn
