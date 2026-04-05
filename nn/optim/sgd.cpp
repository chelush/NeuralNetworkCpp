#include "nn/optim/sgd.hpp"

namespace nn {

Network SGD::apply_gradient_step(float learning_rate, Network net) {
    net.apply_gradients(learning_rate);
    return net;
}

}  // namespace nn
