#pragma once

#include "core/math/Linalg.h"
#include <random>

namespace nn {

class DataLoader;
class MSE;
class Network;
class Optimizer;

struct TrainOptions {
    Eigen::Index epochs;
    Eigen::Index log_every;
};

float train(Network& net, DataLoader& data, MSE& mse, Optimizer& optimizer, float learning_rate,
            const TrainOptions& options, std::mt19937& engine);

}  // namespace nn
