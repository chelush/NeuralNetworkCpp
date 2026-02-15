#include "nn/layers/linear.hpp"
#include "nn/layers/sigmoid.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network.hpp"
#include "nn/optim/sgd.hpp"
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

  nn::MSE mse;
  nn::SGD sgd;
  const float lr = 0.5f;
  const int steps = 50;

  for (int step = 0; step < steps; ++step) {
    nn::MatrixXf pred = net.forward(x);
    float loss = mse.value(pred, y);
    if (step % 10 == 0)
      std::cout << "step " << step << " loss = " << loss << "\n";
    nn::MatrixXf grad = mse.backward(pred, y);
    net.backward(grad);
    sgd.step(net, lr);
  }
  nn::MatrixXf pred = net.forward(x);
  std::cout << "final pred = " << pred(0, 0) << ", y = " << y(0, 0) << "\n";
  return 0;
}
