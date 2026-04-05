#pragma once

#include "core/math/Linalg.h"
#include "core/random/random.hpp"
#include "nn/layers/layer.hpp"
#include "nn/model/network.hpp"
#include <vector>

namespace nn {

class NetworkBuilder {
public:
    static NetworkBuilder set_input(In n, Random& rnd);

    NetworkBuilder& add_linear(Out m);
    /// Последний Linear перед потерей: включает деление градиента на число скаляров (как mean MSE).
    NetworkBuilder& add_output_linear(Out m);
    NetworkBuilder& add_sigmoid();

    Network extract();

private:
    explicit NetworkBuilder(In n, Random& rnd);

    Random& rnd_;
    Eigen::Index previous_;
    std::vector<Layer> layers_;
};

}  // namespace nn
