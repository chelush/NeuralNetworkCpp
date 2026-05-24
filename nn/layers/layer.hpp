#pragma once

#include "core/math/Linalg.h"
#include <memory>

namespace nn {

class Activation;
class Linear;

class Layer {
    struct Concept;
    template <class T>
    struct Model;

    std::unique_ptr<Concept> model_;

public:
    Layer(Activation&& activation);
    Layer(Linear&& layer);

    Layer(Layer&& other) noexcept;
    Layer& operator=(Layer&& other) noexcept;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    ~Layer();

    MatrixXf forward(MatrixXf&& activations);
    MatrixXf backward(MatrixXf&& grad);
    void zero_gradients();
    void clear_cache();
    Linear* as_linear();
    const Linear* as_linear() const;
};

}  // namespace nn
