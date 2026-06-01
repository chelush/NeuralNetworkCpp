#include "nn/loss/mse.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

TEST(MseTest, ValueMatchesFormula) {
    nn::MSE mse;
    nn::MatrixXf pred(1, 3);
    pred << 1.0f, 2.0f, 3.0f;
    nn::MatrixXf target(1, 3);
    target << 0.0f, 2.0f, 5.0f;
    // diff = [1, 0, -2] -> sumsq = 5 -> /3
    EXPECT_NEAR(mse.value(pred, target), 5.0f / 3.0f, 1e-6f);
}

TEST(MseTest, BackwardMatchesFormula) {
    nn::MSE mse;
    nn::MatrixXf pred(1, 3);
    pred << 1.0f, 2.0f, 3.0f;
    nn::MatrixXf target(1, 3);
    target << 0.0f, 2.0f, 5.0f;

    const nn::MatrixXf grad = mse.backward(pred, target);
    // grad = 2 * (pred - target) / n
    nn::MatrixXf expected(1, 3);
    expected << 2.0f / 3.0f, 0.0f, -4.0f / 3.0f;
    EXPECT_TRUE(grad.isApprox(expected, 1e-6f));
}

TEST(MseTest, ShapeMismatchThrows) {
    nn::MSE mse;
    EXPECT_THROW(mse.value(nn::MatrixXf::Zero(2, 3), nn::MatrixXf::Zero(3, 2)),
                 std::invalid_argument);
    EXPECT_THROW(mse.backward(nn::MatrixXf::Zero(2, 3), nn::MatrixXf::Zero(3, 2)),
                 std::invalid_argument);
}

TEST(MseTest, EmptyInputThrows) {
    nn::MSE mse;
    EXPECT_THROW(mse.value(nn::MatrixXf(0, 0), nn::MatrixXf(0, 0)), std::invalid_argument);
    EXPECT_THROW(mse.backward(nn::MatrixXf(0, 0), nn::MatrixXf(0, 0)), std::invalid_argument);
}
