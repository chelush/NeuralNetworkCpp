#include "core/random/random.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network_builder.hpp"
#include "nn/optim/adam.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <stdexcept>

TEST(AdamTest, RejectsInvalidHyperparameters) {
    EXPECT_THROW(nn::Adam(-0.1f, 0.999f, 1e-8f), std::invalid_argument);
    EXPECT_THROW(nn::Adam(0.9f, 1.0f, 1e-8f), std::invalid_argument);
    EXPECT_THROW(nn::Adam(0.9f, 0.999f, 0.0f), std::invalid_argument);
}

TEST(AdamTest, FirstStepMatchesAnalyticalFormula) {
    // y = 2x + 0; one Linear(1 -> 1); manual Adam state after one step.
    std::mt19937 engine(0);
    nn::Random rnd(engine);
    nn::Network net =
        nn::NetworkBuilder::set_input(nn::In{1}, rnd).add_linear(nn::Out{1}).extract();

    const float w0 = net.linear_layers().front()->weights()(0, 0);
    const float b0 = net.linear_layers().front()->bias()(0);

    nn::MatrixXf x(1, 1);
    x << 1.0f;
    nn::MatrixXf y(1, 1);
    y << 2.0f;

    nn::MSE mse;
    nn::MatrixXf pred = net.forward(nn::MatrixXf(x));
    net.backward(mse.backward(pred, y));

    const float dw = net.linear_layers().front()->grad_weights()(0, 0);
    const float db = net.linear_layers().front()->grad_bias()(0);

    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float lr = 0.01f;

    nn::Adam adam(beta1, beta2, eps);
    adam.step(net, lr);

    const float m_w = (1.0f - beta1) * dw;
    const float v_w = (1.0f - beta2) * dw * dw;
    const float m_w_hat = m_w / (1.0f - beta1);
    const float v_w_hat = v_w / (1.0f - beta2);
    const float w_expected = w0 - lr * m_w_hat / (std::sqrt(v_w_hat) + eps);

    const float m_b = (1.0f - beta1) * db;
    const float v_b = (1.0f - beta2) * db * db;
    const float m_b_hat = m_b / (1.0f - beta1);
    const float v_b_hat = v_b / (1.0f - beta2);
    const float b_expected = b0 - lr * m_b_hat / (std::sqrt(v_b_hat) + eps);

    EXPECT_NEAR(net.linear_layers().front()->weights()(0, 0), w_expected, 1e-5f);
    EXPECT_NEAR(net.linear_layers().front()->bias()(0), b_expected, 1e-5f);
}

TEST(AdamTest, DrivesLossToZeroOnLinearProblem) {
    std::mt19937 engine(42);
    nn::Random rnd(engine);
    nn::Network net =
        nn::NetworkBuilder::set_input(nn::In{1}, rnd).add_linear(nn::Out{1}).extract();

    nn::MatrixXf x(8, 1);
    for (int i = 0; i < 8; ++i)
        x(i, 0) = static_cast<float>(i) - 3.5f;
    nn::MatrixXf y = 3.0f * x.array() - 1.0f;

    nn::MSE mse;
    nn::Adam adam;

    const float loss0 = mse.value(net.forward(nn::MatrixXf(x)), y);
    for (int step = 0; step < 800; ++step) {
        nn::MatrixXf pred = net.forward(nn::MatrixXf(x));
        net.backward(mse.backward(pred, y));
        adam.step(net, 0.05f);
        net.zero_gradients();
    }
    const float loss1 = mse.value(net.forward(nn::MatrixXf(x)), y);

    EXPECT_LT(loss1, loss0 * 1e-3f);
    EXPECT_LT(loss1, 1e-4f);
}
