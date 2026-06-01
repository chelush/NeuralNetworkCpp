#pragma once

#include "core/math/Linalg.h"
#include "nn/optim/optimizer.hpp"
#include <vector>

namespace nn {

class Adam final : public Optimizer {
public:
    Adam(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    void step(Network& net, float learning_rate) override;

private:
    struct State {
        MatrixXf mW;
        MatrixXf vW;
        VectorXf mb;
        VectorXf vb;
    };

    float beta1_;
    float beta2_;
    float epsilon_;
    Eigen::Index t_;
    float beta1_pow_;
    float beta2_pow_;
    std::vector<State> states_;
};

}  // namespace nn
