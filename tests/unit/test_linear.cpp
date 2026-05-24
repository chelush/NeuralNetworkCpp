#include "core/random/random.hpp"
#include "nn/layers/linear.hpp"
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>

namespace {

nn::Linear make_layer(Eigen::Index in, Eigen::Index out, std::mt19937& engine) {
    nn::Random rnd(engine);
    return nn::Linear(nn::In{in}, nn::Out{out}, rnd);
}

}  // namespace

TEST(LinearTest, ForwardMatchesAffine) {
    std::mt19937 engine(123);
    auto layer = make_layer(3, 2, engine);

    nn::MatrixXf x(2, 3);
    x << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f;

    const nn::MatrixXf y = layer.forward(nn::MatrixXf(x));
    const nn::MatrixXf expected = (x * layer.weights().transpose()).rowwise() + layer.bias().transpose();

    ASSERT_EQ(y.rows(), 2);
    ASSERT_EQ(y.cols(), 2);
    EXPECT_TRUE(y.isApprox(expected, 1e-6f));
}

TEST(LinearTest, BackwardShapesAndDxFormula) {
    std::mt19937 engine(7);
    auto layer = make_layer(4, 3, engine);

    nn::MatrixXf x(5, 4);
    for (int i = 0; i < x.size(); ++i)
        x.data()[i] = static_cast<float>(i % 7) - 3.0f;

    nn::MatrixXf grad(5, 3);
    grad.setOnes();

    layer.forward(nn::MatrixXf(x));
    const nn::MatrixXf dx = layer.backward(nn::MatrixXf(grad));

    ASSERT_EQ(dx.rows(), x.rows());
    ASSERT_EQ(dx.cols(), x.cols());

    const nn::MatrixXf expected_dx = grad * layer.weights();
    EXPECT_TRUE(dx.isApprox(expected_dx, 1e-5f));

    const nn::MatrixXf expected_dW = grad.transpose() * x;
    EXPECT_TRUE(layer.grad_weights().isApprox(expected_dW, 1e-5f));

    const nn::VectorXf expected_db = grad.colwise().sum().transpose();
    EXPECT_TRUE(layer.grad_bias().isApprox(expected_db, 1e-5f));
}

TEST(LinearTest, ApplyParameterStep) {
    std::mt19937 engine(11);
    auto layer = make_layer(2, 2, engine);

    nn::MatrixXf w_before = layer.weights();
    nn::VectorXf b_before = layer.bias();

    nn::MatrixXf dW = nn::MatrixXf::Ones(2, 2);
    nn::VectorXf db = nn::VectorXf::Ones(2);

    layer.apply_parameter_step(dW, db);

    EXPECT_TRUE(layer.weights().isApprox(w_before + dW, 1e-6f));
    EXPECT_TRUE(layer.bias().isApprox(b_before + db, 1e-6f));
}

TEST(LinearTest, RejectsShapeMismatchOnForward) {
    std::mt19937 engine(1);
    auto layer = make_layer(3, 2, engine);
    nn::MatrixXf bad(2, 5);
    bad.setZero();
    EXPECT_THROW(layer.forward(std::move(bad)), std::invalid_argument);
}

TEST(LinearTest, BackwardWithoutForwardThrows) {
    std::mt19937 engine(2);
    auto layer = make_layer(3, 2, engine);
    nn::MatrixXf grad(2, 2);
    grad.setZero();
    EXPECT_THROW(layer.backward(std::move(grad)), std::logic_error);
}

TEST(LinearTest, ApplyParameterStepShapeMismatchThrows) {
    std::mt19937 engine(3);
    auto layer = make_layer(3, 2, engine);
    EXPECT_THROW(layer.apply_parameter_step(nn::MatrixXf::Zero(3, 3), nn::VectorXf::Zero(2)),
                 std::invalid_argument);
    EXPECT_THROW(layer.apply_parameter_step(nn::MatrixXf::Zero(2, 3), nn::VectorXf::Zero(5)),
                 std::invalid_argument);
}
