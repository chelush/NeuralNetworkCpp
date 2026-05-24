#include "nn/layers/layer.hpp"
#include "nn/layers/activation.hpp"
#include "nn/layers/linear.hpp"
#include <type_traits>

namespace nn {

struct Layer::Concept {
    virtual ~Concept() = default;
    virtual MatrixXf forward(MatrixXf&& activations) = 0;
    virtual MatrixXf backward(MatrixXf&& grad) = 0;
    virtual void zero_gradients() = 0;
    virtual void clear_cache() = 0;
    virtual Linear* as_linear() = 0;
    virtual const Linear* as_linear() const = 0;
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
    void zero_gradients() final {
        data_.zero_gradients();
    }
    void clear_cache() final {
        data_.clear_cache();
    }
    Linear* as_linear() final {
        if constexpr (std::is_same_v<T, Linear>)
            return &data_;
        return nullptr;
    }
    const Linear* as_linear() const final {
        if constexpr (std::is_same_v<T, Linear>)
            return &data_;
        return nullptr;
    }
};

Layer::Layer(Activation&& activation)
    : model_(std::make_unique<Model<Activation>>(std::move(activation))) {
}

Layer::Layer(Linear&& layer) : model_(std::make_unique<Model<Linear>>(std::move(layer))) {
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

void Layer::zero_gradients() {
    model_->zero_gradients();
}

void Layer::clear_cache() {
    model_->clear_cache();
}

Linear* Layer::as_linear() {
    return model_->as_linear();
}

const Linear* Layer::as_linear() const {
    return model_->as_linear();
}

}  // namespace nn
