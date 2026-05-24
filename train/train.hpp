#pragma once

#include "core/math/Linalg.h"
#include <random>

namespace nn {

class DataLoader;
class Loss;
class LRSchedule;
class Network;
class Optimizer;

struct TrainOptions {
    Eigen::Index epochs;
    Eigen::Index log_every;
};

float train(Network& net, DataLoader& data, Loss& loss, Optimizer& optimizer,
            const LRSchedule& schedule, const TrainOptions& options, std::mt19937& engine);

float train(Network& net, DataLoader& data, Loss& loss, Optimizer& optimizer, float learning_rate,
            const TrainOptions& options, std::mt19937& engine);

}  // namespace nn
