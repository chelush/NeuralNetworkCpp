#include "train/train.hpp"
#include "app/except.h"
#include "nn/loss/loss.hpp"
#include "nn/model/network.hpp"
#include "nn/scheduler/lr_schedule.hpp"
#include "nn/optim/optimizer.hpp"
#include "utils/data_loader.hpp"
#include <iostream>
#include <utility>

namespace nn {

float train(Network& net, DataLoader& data, Loss& loss, Optimizer& optimizer,
            const LRSchedule& schedule, const TrainOptions& options, std::mt19937& engine) {
    NN_REQUIRE(data.size() > 0, "train(): dataset must contain at least one sample");
    NN_REQUIRE(options.epochs > 0, "train(): epochs must be > 0");

    float epoch_loss = 0.0f;
    for (Eigen::Index epoch = 0; epoch < options.epochs; ++epoch) {
        data.reset_epoch(engine);

        const float lr = schedule.at(epoch);
        NN_REQUIRE(lr > 0.0f, "train(): LRSchedule produced non-positive lr");

        float weighted_loss = 0.0f;
        Eigen::Index seen = 0;

        for (Batch batch : data) {
            MatrixXf pred = net.forward(std::move(batch.X));
            float batch_loss = loss.value(pred, batch.Y);
            weighted_loss += batch_loss * batch.Y.rows();
            seen += batch.Y.rows();

            net.backward(loss.backward(pred, batch.Y));
            optimizer.step(net, lr);
            net.zero_gradients();
        }

        NN_ASSERT(seen > 0, "train(): internal invariant broken, epoch has no samples");
        epoch_loss = weighted_loss / seen;
        if (options.log_every > 0 && epoch % options.log_every == 0)
            std::cout << "epoch " << epoch << " lr = " << lr << " loss = " << epoch_loss << "\n";
    }

    return epoch_loss;
}

float train(Network& net, DataLoader& data, Loss& loss, Optimizer& optimizer, float learning_rate,
            const TrainOptions& options, std::mt19937& engine) {
    const ConstantSchedule schedule(learning_rate);
    return train(net, data, loss, optimizer, schedule, options, engine);
}

}  // namespace nn
