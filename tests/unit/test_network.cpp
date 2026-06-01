#include "core/random/random.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network_builder.hpp"
#include "nn/optim/adam.hpp"
#include <gtest/gtest.h>
#include <random>

TEST(NetworkTest, ForwardOutputShapeMatchesBuilder) {
    std::mt19937 engine(0);
    nn::Random rnd(engine);

    nn::Network net = nn::NetworkBuilder::set_input(nn::In{3}, rnd)
                          .add_linear(nn::Out{5})
                          .add_relu()
                          .add_linear(nn::Out{2})
                          .extract();

    nn::MatrixXf x = nn::MatrixXf::Random(4, 3);
    const nn::MatrixXf y = net.forward(std::move(x));
    EXPECT_EQ(y.rows(), 4);
    EXPECT_EQ(y.cols(), 2);
}

TEST(NetworkTest, LinearLayersCacheIsStableAndOnlyLinear) {
    std::mt19937 engine(7);
    nn::Random rnd(engine);

    nn::Network net = nn::NetworkBuilder::set_input(nn::In{2}, rnd)
                          .add_linear(nn::Out{4})
                          .add_relu()
                          .add_linear(nn::Out{1})
                          .extract();

    const auto& cache1 = net.linear_layers();
    const auto& cache2 = net.linear_layers();

    ASSERT_EQ(cache1.size(), 2u);
    EXPECT_EQ(&cache1, &cache2);
    EXPECT_EQ(cache1[0], cache2[0]);
    EXPECT_EQ(cache1[1], cache2[1]);

    EXPECT_NE(cache1[0], nullptr);
    EXPECT_NE(cache1[1], nullptr);
}

TEST(NetworkTest, MlpFitsXorRoughlyWithAdam) {
    std::mt19937 engine(123);
    nn::Random rnd(engine);

    nn::Network net = nn::NetworkBuilder::set_input(nn::In{2}, rnd)
                          .add_linear(nn::Out{8})
                          .add_tanh()
                          .add_linear(nn::Out{1})
                          .extract();

    nn::MatrixXf X(4, 2);
    X << 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f;
    nn::MatrixXf Y(4, 1);
    Y << 0.0f, 1.0f, 1.0f, 0.0f;

    nn::MSE mse;
    nn::Adam adam;

    const float loss0 = mse.value(net.forward(nn::MatrixXf(X)), Y);
    for (int step = 0; step < 4000; ++step) {
        nn::MatrixXf pred = net.forward(nn::MatrixXf(X));
        net.backward(mse.backward(pred, Y));
        adam.step(net, 0.05f);
        net.zero_gradients();
    }
    const float loss1 = mse.value(net.forward(nn::MatrixXf(X)), Y);

    EXPECT_LT(loss1, loss0 * 0.1f);
    EXPECT_LT(loss1, 0.05f);
}
