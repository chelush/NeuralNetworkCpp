#pragma once

#include "core/math/types.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/sigmoid.hpp"
#include <functional>
#include <variant>
#include <vector>

namespace nn {

using LayerVariant = std::variant<Linear, Sigmoid>;

class Network {
public:
  void add(Linear layer);
  void add(Sigmoid layer);
  MatrixXf forward(const MatrixXf& x);
  MatrixXf backward(const MatrixXf& grad_out);
  void for_each_linear(std::function<void(Linear&)> f);

private:
  std::vector<LayerVariant> layers_;
};

}  // namespace nn
