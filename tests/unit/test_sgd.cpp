#include "core/random/random.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network_builder.hpp"
#include "nn/optim/sgd.hpp"
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>

TEST(SgdTest, StepDecreasesLossOnLinearProblem) {
    std::mt19937 engine(42);
    nn::Random rnd(engine);

    nn::Network net =
        nn::NetworkBuilder::set_input(nn::In{1}, rnd).add_linear(nn::Out{1}).extract();

    nn::MatrixXf x(4, 1);
    x << -1.0f, 0.0f, 1.0f, 2.0f;
    nn::MatrixXf y(4, 1);
    y << -2.0f, 0.0f, 2.0f, 4.0f;

    nn::MSE mse;
    nn::SGD sgd;

    nn::MatrixXf pred0 = net.forward(nn::MatrixXf(x));
    const float loss0 = mse.value(pred0, y);

    for (int step = 0; step < 200; ++step) {
        nn::MatrixXf pred = net.forward(nn::MatrixXf(x));
        net.backward(mse.backward(pred, y));
        sgd.step(net, 0.05f);
        net.zero_gradients();
    }

    nn::MatrixXf pred1 = net.forward(nn::MatrixXf(x));
    const float loss1 = mse.value(pred1, y);

    EXPECT_LT(loss1, loss0 * 0.1f);
    EXPECT_LT(loss1, 1e-3f);
}

TEST(SgdTest, RejectsNonPositiveLearningRate) {
    std::mt19937 engine(1);
    nn::Random rnd(engine);
    nn::Network net =
        nn::NetworkBuilder::set_input(nn::In{1}, rnd).add_linear(nn::Out{1}).extract();
    nn::SGD sgd;
    EXPECT_THROW(sgd.step(net, 0.0f), std::invalid_argument);
    EXPECT_THROW(sgd.step(net, -1.0f), std::invalid_argument);
}
