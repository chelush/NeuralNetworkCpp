#include "nn/optim/sgd.hpp"
#include "nn/layers/linear.hpp"

namespace nn {

void SGD::step(Network& net, float lr) {
  net.for_each_linear([lr](Linear& linear) {
    linear.weights() -= lr * linear.grad_weights();
    linear.bias() -= lr * linear.grad_bias();
  });
}

}  // namespace nn
