#include "train/train.hpp"
#include "app/except.h"
#include "nn/loss/mse.hpp"
#include "nn/model/network.hpp"
#include "nn/optim/optimizer.hpp"
#include "utils/data_loader.hpp"
#include <iostream>
#include <utility>

namespace nn {

float train(Network& net, DataLoader& data, MSE& mse, Optimizer& optimizer, float learning_rate,
            const TrainOptions& options, std::mt19937& engine) {
    NN_REQUIRE(data.size() > 0, "train(): dataset must contain at least one sample");
    NN_REQUIRE(options.epochs > 0, "train(): epochs must be > 0");
    NN_REQUIRE(learning_rate > 0.0f, "train(): learning_rate must be > 0");

    float epoch_loss = 0.0f;
    for (Eigen::Index epoch = 0; epoch < options.epochs; ++epoch) {
        data.reset_epoch(engine);

        float weighted_loss = 0.0f;
        Eigen::Index seen = 0;

        for (Batch batch : data) {
            MatrixXf pred = net.forward(std::move(batch.X));
            float loss = mse.value(pred, batch.Y);
            weighted_loss += loss * batch.Y.rows();
            seen += batch.Y.rows();

            net.backward(mse.backward(pred, batch.Y));
            optimizer.step(net, learning_rate);
            net.zero_gradients();
        }

        NN_ASSERT(seen > 0, "train(): internal invariant broken, epoch has no samples");
        epoch_loss = weighted_loss / seen;
        if (options.log_every > 0 && epoch % options.log_every == 0)
            std::cout << "epoch " << epoch << " loss = " << epoch_loss << "\n";
    }

    return epoch_loss;
}

}  // namespace nn
