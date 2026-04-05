#include "nn/layers/layer.hpp"
#include "nn/layers/linear.hpp"
#include "nn/layers/sigmoid.hpp"

namespace nn {

struct Layer::Concept {
    virtual ~Concept() = default;
    virtual MatrixXf forward(MatrixXf&& activations) = 0;
    virtual MatrixXf backward(MatrixXf&& grad) = 0;
    virtual void apply_gradients(float learning_rate) = 0;
    virtual void zero_gradients() = 0;
    virtual void clear_cache() = 0;
};

template <class T>
struct Layer::Model final : Layer::Concept {
    T data_;

    explicit Model(T&& data) : data_(std::move(data)) {
    }

    MatrixXf forward(MatrixXf&& activations) final {
        return data_.forward(std::move(activations));
    }
    MatrixXf backward(MatrixXf&& grad) final {
        return data_.backward(std::move(grad));
    }
    void apply_gradients(float learning_rate) final {
        data_.apply_gradients(learning_rate);
    }
    void zero_gradients() final {
        data_.zero_gradients();
    }
    void clear_cache() final {
        data_.clear_cache();
    }
};

Layer::Layer(Linear&& layer) : model_(std::make_unique<Model<Linear>>(std::move(layer))) {
}

Layer::Layer(Sigmoid&& activation)
    : model_(std::make_unique<Model<Sigmoid>>(std::move(activation))) {
}

Layer::Layer(Layer&&) noexcept = default;

Layer& Layer::operator=(Layer&&) noexcept = default;

Layer::~Layer() = default;

MatrixXf Layer::forward(MatrixXf&& activations) {
    return model_->forward(std::move(activations));
}

MatrixXf Layer::backward(MatrixXf&& grad) {
    return model_->backward(std::move(grad));
}

void Layer::apply_gradients(float learning_rate) {
    model_->apply_gradients(learning_rate);
}

void Layer::zero_gradients() {
    model_->zero_gradients();
}

void Layer::clear_cache() {
    model_->clear_cache();
}

}  // namespace nn
