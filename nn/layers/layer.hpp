#pragma once

#include "core/math/Linalg.h"
#include <memory>

namespace nn {

class Linear;
class Sigmoid;

class Layer {
    struct Concept;
    template <class T>
    struct Model;

    std::unique_ptr<Concept> model_;

public:
    Layer(Linear&& layer);
    Layer(Sigmoid&& activation);

    Layer(Layer&& other) noexcept;
    Layer& operator=(Layer&& other) noexcept;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    ~Layer();

    MatrixXf forward(MatrixXf&& activations);
    MatrixXf backward(MatrixXf&& grad);
    void apply_gradients(float learning_rate);
    void zero_gradients();
    void clear_cache();
};

}  // namespace nn
