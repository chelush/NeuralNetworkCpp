#include "nn/optim/sgd.hpp"
#include "nn/layers/linear.hpp"

namespace nn {

void SGD::step(Network& net, float lr) {
  for (Layer* l : net.layers()) {
    auto* linear = dynamic_cast<Linear*>(l);
    if (linear) {
      linear->weights() -= lr * linear->grad_weights();
      linear->bias() -= lr * linear->grad_bias();
    }
  }
}

}  // namespace nn
