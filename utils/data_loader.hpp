#pragma once

#include "core/math/Linalg.h"
#include <cstddef>
#include <random>
#include <vector>

namespace nn {

struct Batch {
    MatrixXf X;
    MatrixXf Y;
};

struct BatchSize {
    std::size_t value;
};

struct Shuffle {
    bool value;
};

class DataLoader {
public:
    DataLoader(MatrixXf X, MatrixXf Y, BatchSize batch_size, Shuffle shuffle = Shuffle{true});

    void reset_epoch();
    void reset_epoch(std::mt19937& engine);

    std::size_t num_batches() const;
    std::size_t batch_size() const;
    std::size_t size() const;

    Batch get_batch(std::size_t batch_idx) const;

    class Iterator {
    public:
        Iterator(const DataLoader* loader, std::size_t idx);

        Batch operator*() const;
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;

    private:
        const DataLoader* loader_;
        std::size_t idx_;
    };

    Iterator begin() const;
    Iterator end() const;

private:
    MatrixXf X_;
    MatrixXf Y_;
    std::size_t batch_size_;
    bool shuffle_;
    std::size_t epoch_ = 0;
    std::vector<Eigen::Index> indices_;
};

}  // namespace nn
