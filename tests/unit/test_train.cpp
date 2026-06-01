#include "core/random/random.hpp"
#include "nn/loss/bce.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network_builder.hpp"
#include "nn/optim/adam.hpp"
#include "nn/scheduler/lr_schedule.hpp"
#include "nn/optim/sgd.hpp"
#include "train/train.hpp"
#include "utils/data_loader.hpp"
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>

namespace {

nn::Network make_linear_net(std::mt19937& engine) {
    nn::Random rnd(engine);
    return nn::NetworkBuilder::set_input(nn::In{1}, rnd).add_linear(nn::Out{1}).extract();
}

}  // namespace

TEST(TrainTest, WorksWithMseLoss) {
    std::mt19937 engine(42);
    auto net = make_linear_net(engine);

    nn::MatrixXf X(16, 1);
    for (int i = 0; i < 16; ++i)
        X(i, 0) = static_cast<float>(i) - 7.5f;
    nn::MatrixXf Y = 2.0f * X.array() + 1.0f;

    nn::DataLoader loader(std::move(X), std::move(Y), nn::BatchSize{4}, nn::Shuffle{true});
    nn::MSE mse;
    nn::Adam adam;

    const float loss = nn::train(net, loader, mse, adam, 0.05f, nn::TrainOptions{50, 0}, engine);
    EXPECT_LT(loss, 1e-3f);
}

TEST(TrainTest, WorksWithBceLoss) {
    std::mt19937 engine(1);
    nn::Random rnd(engine);
    nn::Network net = nn::NetworkBuilder::set_input(nn::In{2}, rnd)
                          .add_linear(nn::Out{8})
                          .add_tanh()
                          .add_linear(nn::Out{1})
                          .add_sigmoid()
                          .extract();

    nn::MatrixXf X(4, 2);
    X << 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f;
    nn::MatrixXf Y(4, 1);
    Y << 0.0f, 1.0f, 1.0f, 0.0f;

    nn::DataLoader loader(std::move(X), std::move(Y), nn::BatchSize{4}, nn::Shuffle{true});
    nn::BCE bce;
    nn::Adam adam;

    const float loss0 =
        bce.value(net.forward(nn::MatrixXf(loader.get_batch(0).X)), loader.get_batch(0).Y);
    const float loss =
        nn::train(net, loader, bce, adam, 0.05f, nn::TrainOptions{2000, 0}, engine);

    EXPECT_LT(loss, loss0 * 0.5f);
    EXPECT_LT(loss, 0.3f);
}

TEST(TrainTest, RejectsBadOptions) {
    std::mt19937 engine(0);
    auto net = make_linear_net(engine);
    nn::DataLoader loader(nn::MatrixXf::Zero(4, 1), nn::MatrixXf::Zero(4, 1), nn::BatchSize{2},
                          nn::Shuffle{false});
    nn::MSE mse;
    nn::SGD sgd;

    EXPECT_THROW(nn::train(net, loader, mse, sgd, 0.01f, nn::TrainOptions{0, 0}, engine),
                 std::invalid_argument);
    EXPECT_THROW(nn::train(net, loader, mse, sgd, 0.0f, nn::TrainOptions{1, 0}, engine),
                 std::invalid_argument);
}

TEST(TrainTest, WorksWithCosineAnnealingSchedule) {
    std::mt19937 engine(7);
    auto net = make_linear_net(engine);

    nn::MatrixXf X(16, 1);
    for (int i = 0; i < 16; ++i)
        X(i, 0) = static_cast<float>(i) - 7.5f;
    nn::MatrixXf Y = -3.0f * X.array() + 0.5f;

    nn::DataLoader loader(std::move(X), std::move(Y), nn::BatchSize{4}, nn::Shuffle{true});
    nn::MSE mse;
    nn::Adam adam;
    nn::CosineAnnealingSchedule schedule(0.1f, 1e-4f, 50);

    const float loss =
        nn::train(net, loader, mse, adam, schedule, nn::TrainOptions{50, 0}, engine);
    EXPECT_LT(loss, 1e-3f);
}

TEST(TrainTest, StepDecayChangesLearningRateOverEpochs) {
    // Sanity-check that schedule values change vs flat lr: train with strong decay
    // and observe parameters move less than with constant high lr.
    std::mt19937 engine(11);
    auto net = make_linear_net(engine);

    nn::MatrixXf X(8, 1);
    for (int i = 0; i < 8; ++i)
        X(i, 0) = static_cast<float>(i);
    nn::MatrixXf Y = 5.0f * X.array() + 1.0f;

    nn::DataLoader loader_a(nn::MatrixXf(X), nn::MatrixXf(Y), nn::BatchSize{2}, nn::Shuffle{false});
    nn::DataLoader loader_b(nn::MatrixXf(X), nn::MatrixXf(Y), nn::BatchSize{2}, nn::Shuffle{false});

    nn::MSE mse;
    nn::Adam adam_const;
    nn::Adam adam_decay;
    nn::StepDecaySchedule decay(0.1f, 1, 0.01f);

    const float loss_const =
        nn::train(net, loader_a, mse, adam_const, 0.1f, nn::TrainOptions{5, 0}, engine);
    auto net2 = make_linear_net(engine);
    const float loss_decay =
        nn::train(net2, loader_b, mse, adam_decay, decay, nn::TrainOptions{5, 0}, engine);

    // After 5 epochs the constant-lr run should be at a strictly smaller loss than
    // the aggressively decayed run.
    EXPECT_LT(loss_const, loss_decay);
}
