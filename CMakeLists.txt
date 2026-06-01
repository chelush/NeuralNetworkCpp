cmake_minimum_required(VERSION 3.16)
project(NeuralNetwork VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Архитектура (по ТЗ):
#   core/math    — Linalg.h
#   nn/layers    — Linear + активации
#   nn/loss      — MSE / BCE
#   nn/optim     — SGD / Adam
#   nn/scheduler — LR scheduler-ы
#   nn/model     — Network, управление слоями
#   train        — цикл обучения, батчи, метрики
#   app          — CLI / демо

# Eigen (header-only)
include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
set(BUILD_TESTING OFF CACHE BOOL "Disable dependency tests" FORCE)
FetchContent_MakeAvailable(Eigen)

set(SOURCES
  app/except.cpp
  core/random/random.cpp
  train/train.cpp
  utils/data_loader.cpp
  nn/layers/activation.cpp
  nn/layers/layer.cpp
  nn/layers/linear.cpp
  nn/loss/bce.cpp
  nn/loss/mse.cpp
  nn/model/network.cpp
  nn/model/network_builder.cpp
  nn/optim/adam.cpp
  nn/optim/sgd.cpp
  nn/scheduler/lr_schedule.cpp
)

add_library(neuralnetwork STATIC ${SOURCES})
target_include_directories(neuralnetwork PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(neuralnetwork PUBLIC Eigen3::Eigen)
if (MSVC)
  target_compile_options(neuralnetwork PRIVATE /W4)
else()
  target_compile_options(neuralnetwork PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(demo app/main.cpp)
target_link_libraries(demo PRIVATE neuralnetwork)

option(NN_BUILD_TESTS "Build unit tests" ON)
if (NN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
