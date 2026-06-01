#include "nn/layers/activation.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

TEST(ActivationTest, ReluForwardAndBackward) {
    nn::Activation relu(nn::ActivationKind::ReLU);
    nn::MatrixXf x(2, 2);
    x << -1.0f, 2.0f, 0.0f, -3.0f;

    const nn::MatrixXf y = relu.forward(nn::MatrixXf(x));
    nn::MatrixXf expected_y(2, 2);
    expected_y << 0.0f, 2.0f, 0.0f, 0.0f;
    EXPECT_TRUE(y.isApprox(expected_y, 1e-6f));

    nn::MatrixXf grad(2, 2);
    grad << 1.0f, 1.0f, 1.0f, 1.0f;
    const nn::MatrixXf dx = relu.backward(nn::MatrixXf(grad));
    nn::MatrixXf expected_dx(2, 2);
    expected_dx << 0.0f, 1.0f, 0.0f, 0.0f;
    EXPECT_TRUE(dx.isApprox(expected_dx, 1e-6f));
}

TEST(ActivationTest, SigmoidForwardInRangeAndBackward) {
    nn::Activation sigmoid(nn::ActivationKind::Sigmoid);
    nn::MatrixXf x(1, 3);
    x << -10.0f, 0.0f, 10.0f;

    const nn::MatrixXf y = sigmoid.forward(nn::MatrixXf(x));
    EXPECT_NEAR(y(0, 0), 0.0f, 1e-3f);
    EXPECT_NEAR(y(0, 1), 0.5f, 1e-6f);
    EXPECT_NEAR(y(0, 2), 1.0f, 1e-3f);

    nn::MatrixXf grad(1, 3);
    grad << 1.0f, 2.0f, 3.0f;
    const nn::MatrixXf dx = sigmoid.backward(nn::MatrixXf(grad));
    for (int j = 0; j < 3; ++j) {
        const float expected = grad(0, j) * y(0, j) * (1.0f - y(0, j));
        EXPECT_NEAR(dx(0, j), expected, 1e-6f);
    }
}

TEST(ActivationTest, TanhForwardAndBackward) {
    nn::Activation tanh_layer(nn::ActivationKind::Tanh);
    nn::MatrixXf x(1, 3);
    x << -1.0f, 0.0f, 1.0f;

    const nn::MatrixXf y = tanh_layer.forward(nn::MatrixXf(x));
    EXPECT_NEAR(y(0, 0), std::tanh(-1.0f), 1e-6f);
    EXPECT_NEAR(y(0, 1), 0.0f, 1e-6f);
    EXPECT_NEAR(y(0, 2), std::tanh(1.0f), 1e-6f);

    nn::MatrixXf grad(1, 3);
    grad.setOnes();
    const nn::MatrixXf dx = tanh_layer.backward(nn::MatrixXf(grad));
    for (int j = 0; j < 3; ++j) {
        const float expected = 1.0f - y(0, j) * y(0, j);
        EXPECT_NEAR(dx(0, j), expected, 1e-6f);
    }
}

TEST(ActivationTest, BackwardWithoutForwardThrows) {
    nn::Activation relu(nn::ActivationKind::ReLU);
    nn::MatrixXf grad(2, 2);
    grad.setZero();
    EXPECT_THROW(relu.backward(std::move(grad)), std::logic_error);
}
