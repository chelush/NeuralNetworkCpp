#pragma once

#include "core/math/Linalg.h"
#include <random>

namespace nn {

class Random {
public:
    explicit Random(std::mt19937& engine);

    MatrixXf uniform_matrix(Out out, In in);
    VectorXf uniform_vector(Out out);

private:
    void fill_uniform_(float* ptr, Eigen::Index count);

    std::mt19937& engine_;
};

}  // namespace nn
