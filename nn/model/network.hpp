#pragma once

#include "core/math/types.hpp"
#include "nn/layers/layer.hpp"
#include <memory>
#include <vector>

namespace nn {

class Network {
public:
  void add(std::unique_ptr<Layer> layer);
  MatrixXf forward(const MatrixXf& x);
  MatrixXf backward(const MatrixXf& grad_out);
  std::vector<Layer*>& layers() { return layer_ptrs_; }

private:
  std::vector<std::unique_ptr<Layer>> layers_;
  std::vector<Layer*> layer_ptrs_;
  std::vector<MatrixXf> activations_;
};

}  // namespace nn
