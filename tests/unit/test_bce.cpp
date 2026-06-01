#include "nn/loss/bce.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

TEST(BceTest, ValueMatchesFormula) {
    nn::BCE bce;
    nn::MatrixXf pred(1, 2);
    pred << 0.7f, 0.2f;
    nn::MatrixXf target(1, 2);
    target << 1.0f, 0.0f;

    const float expected = -(std::log(0.7f) + std::log(0.8f)) / 2.0f;
    EXPECT_NEAR(bce.value(pred, target), expected, 1e-5f);
}

TEST(BceTest, BackwardMatchesFormula) {
    nn::BCE bce;
    nn::MatrixXf pred(1, 2);
    pred << 0.7f, 0.2f;
    nn::MatrixXf target(1, 2);
    target << 1.0f, 0.0f;

    const nn::MatrixXf grad = bce.backward(pred, target);
    // (p - y) / (p*(1-p)*n)
    const float n = 2.0f;
    EXPECT_NEAR(grad(0, 0), (0.7f - 1.0f) / (0.7f * 0.3f * n), 1e-5f);
    EXPECT_NEAR(grad(0, 1), (0.2f - 0.0f) / (0.2f * 0.8f * n), 1e-5f);
}

TEST(BceTest, ClampsExtremeValues) {
    nn::BCE bce;
    nn::MatrixXf pred(1, 2);
    pred << 0.0f, 1.0f;
    nn::MatrixXf target(1, 2);
    target << 1.0f, 0.0f;

    const float loss = bce.value(pred, target);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GT(loss, 0.0f);
}

TEST(BceTest, ShapeMismatchThrows) {
    nn::BCE bce;
    EXPECT_THROW(bce.value(nn::MatrixXf::Zero(1, 2), nn::MatrixXf::Zero(2, 1)),
                 std::invalid_argument);
    EXPECT_THROW(bce.backward(nn::MatrixXf::Zero(1, 2), nn::MatrixXf::Zero(2, 1)),
                 std::invalid_argument);
}

TEST(BceTest, EmptyInputThrows) {
    nn::BCE bce;
    EXPECT_THROW(bce.value(nn::MatrixXf(0, 0), nn::MatrixXf(0, 0)), std::invalid_argument);
    EXPECT_THROW(bce.backward(nn::MatrixXf(0, 0), nn::MatrixXf(0, 0)), std::invalid_argument);
}
