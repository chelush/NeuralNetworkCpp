cmake_minimum_required(VERSION 3.16)
project(NeuralNetwork VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Архитектура (по ТЗ):
#   core/math   — алиасы и утилиты на Eigen
#   nn/layers   — Linear + активации
#   nn/loss     — MSE / BCE
#   nn/optim    — SGD / Adam
#   nn/model    — Network, управление слоями
#   train       — цикл обучения, батчи, метрики
#   app         — CLI / демо

# Eigen (header-only)
include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen)

set(SOURCES
  app/main.cpp
  nn/layers/linear.cpp
  nn/layers/sigmoid.cpp
  nn/loss/mse.cpp
  nn/model/network.cpp
  nn/optim/sgd.cpp
)
add_executable(demo ${SOURCES})
target_include_directories(demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demo PRIVATE Eigen3::Eigen)