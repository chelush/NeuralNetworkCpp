#include "nn/optim/sgd.hpp"
#include "app/except.h"

namespace nn {

void SGD::step(Network& net, float learning_rate) {
    NN_REQUIRE(learning_rate > 0.0f, "SGD::step(): learning_rate must be > 0");
    for (Linear* linear : net.linear_layers()) {
        NN_ASSERT(linear != nullptr, "SGD::step(): null linear layer pointer");
        linear->apply_parameter_step(-learning_rate * linear->grad_weights(),
                                     -learning_rate * linear->grad_bias());
    }
}

}  // namespace nn
