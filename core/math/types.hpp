#pragma once

#include <Eigen/Dense>
#include <cstddef>

namespace nn {

using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;

// Сильные типы для размерностей: не перепутаешь вход и выход.
struct In {
  std::size_t value;
  explicit In(std::size_t n) : value(n) {}
};
struct Out {
  std::size_t value;
  explicit Out(std::size_t n) : value(n) {}
};

}  // namespace nn
