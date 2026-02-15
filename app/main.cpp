#include "nn/layers/linear.hpp"
#include "nn/layers/sigmoid.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network.hpp"
#include <iostream>

int main() {
  nn::Network net;
  net.add(std::make_unique<nn::Linear>(2, 4));
  net.add(std::make_unique<nn::Sigmoid>());
  net.add(std::make_unique<nn::Linear>(4, 1));

  nn::MatrixXf x(1, 2);
  x << 0.5f, -0.3f;
  nn::MatrixXf y(1, 1);
  y << 0.8f;

  nn::MatrixXf pred = net.forward(x);
  nn::MSE mse;
  float loss = mse.value(pred, y);
  std::cout << "pred = " << pred(0, 0) << ", y = " << y(0, 0) << ", loss = " << loss << "\n";
  return 0;
}
