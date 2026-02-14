#include <Eigen/Dense>
#include <iostream>

int main() {
  Eigen::MatrixXf A(2, 2);
  A << 1, 2, 3, 4;
  Eigen::VectorXf v(2);
  v << 1, 0;
  Eigen::VectorXf u = A * v;
  std::cout << "NeuralNetwork (Eigen OK): A*v = [" << u(0) << ", " << u(1) << "]\n";
  return 0;
}
