#pragma once

#include "core/math/Linalg.h"

namespace nn {

class MSE;
class Network;
class SGD;

void train(Network& net, const MatrixXf& inputs, const MatrixXf& targets, MSE& mse, SGD& sgd,
           float learning_rate, Eigen::Index steps, Eigen::Index log_every);

}  // namespace nn
