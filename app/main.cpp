#include "core/random/random.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network_builder.hpp"
#include "nn/optim/sgd.hpp"
#include "train/train.hpp"
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>

using namespace nn;

int main() {
    constexpr Eigen::Index N = 1000;
    std::mt19937 engine(42);
    Random rnd(engine);
    std::uniform_real_distribution<float> dist(-std::numbers::pi_v<float>,
                                               std::numbers::pi_v<float>);

    MatrixXf x(N, 1);
    MatrixXf y(N, 1);
    for (Eigen::Index i = 0; i < N; ++i) {
        float xi = dist(engine);
        x(i, 0) = xi;
        y(i, 0) = std::sin(xi);
    }

    Network net = NetworkBuilder::set_input(In{1}, rnd)
                      .add_linear(Out{32})
                      .add_sigmoid()
                      .add_output_linear(Out{1})
                      .extract();

    MSE mse;
    SGD sgd;
    constexpr float lr = 0.05f;
    constexpr Eigen::Index steps = 500;

    train(net, x, y, mse, sgd, lr, steps, 50);

    MatrixXf pred = net.forward(x);
    std::cout << "sample: x=" << x(0, 0) << " y=" << y(0, 0) << " pred=" << pred(0, 0) << "\n";
    return 0;
}
