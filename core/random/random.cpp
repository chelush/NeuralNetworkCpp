#include "core/random/random.hpp"

nn::Random::Random(std::mt19937& engine) : engine_(engine) {
}

void nn::Random::fill_uniform_(float* ptr, Eigen::Index count) {
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (Eigen::Index i = 0; i < count; ++i)
        ptr[i] = dist(engine_);
}

nn::MatrixXf nn::Random::uniform_matrix(nn::Out out, nn::In in) {
    nn::MatrixXf m(out.value, in.value);
    fill_uniform_(m.data(), m.size());
    return m;
}

nn::VectorXf nn::Random::uniform_vector(nn::Out out) {
    nn::VectorXf v(out.value);
    fill_uniform_(v.data(), v.size());
    return v;
}

nn::Random& nn::GRandom::access() {
    static std::mt19937 engine{42};
    static nn::Random rnd{engine};
    return rnd;
}
