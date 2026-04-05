#include "train/train.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network.hpp"
#include "nn/optim/sgd.hpp"
#include <iostream>
#include <utility>

namespace nn {

void train(Network& net, const MatrixXf& inputs, const MatrixXf& targets, MSE& mse, SGD& sgd,
           float learning_rate, Eigen::Index steps, Eigen::Index log_every) {
    for (Eigen::Index step = 0; step < steps; ++step) {
        MatrixXf pred = net.forward(inputs);
        float loss = mse.value(pred, targets);
        if (log_every > 0 && step % log_every == 0)
            std::cout << "step " << step << " loss = " << loss << "\n";
        net.backward(mse.backward(pred, targets));
        net = sgd.apply_gradient_step(learning_rate, std::move(net));
        net.zero_gradients();
    }
}

}  // namespace nn
