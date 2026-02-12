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

add_executable(demo app/main.cpp)