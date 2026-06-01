#include "utils/data_loader.hpp"
#include "app/except.h"
#include <algorithm>
#include <numeric>
#include <utility>

namespace nn {

DataLoader::DataLoader(MatrixXf X, MatrixXf Y, BatchSize batch_size, Shuffle shuffle)
    : X_(std::move(X)), Y_(std::move(Y)), batch_size_(batch_size.value), shuffle_(shuffle.value) {
    NN_REQUIRE(X_.rows() == Y_.rows(), "DataLoader: X and Y must have equal number of rows");
    NN_REQUIRE(batch_size_ > 0, "DataLoader: batch size must be > 0");

    indices_.resize(static_cast<std::size_t>(X_.rows()));
    std::iota(indices_.begin(), indices_.end(), Eigen::Index{0});
}

void DataLoader::reset_epoch() {
    NN_REQUIRE(!shuffle_, "DataLoader: reset_epoch() without RNG is invalid when shuffle=true");
    ++epoch_;
}

void DataLoader::reset_epoch(std::mt19937& engine) {
    if (shuffle_)
        std::shuffle(indices_.begin(), indices_.end(), engine);
    ++epoch_;
}

std::size_t DataLoader::num_batches() const {
    return (size() + batch_size_ - 1) / batch_size_;
}

std::size_t DataLoader::batch_size() const {
    return batch_size_;
}

std::size_t DataLoader::size() const {
    return indices_.size();
}

Batch DataLoader::get_batch(std::size_t batch_idx) const {
    NN_REQUIRE(batch_idx < num_batches(), "DataLoader: batch index is out of range");
    const std::size_t start = batch_idx * batch_size_;
    const std::size_t end = std::min(start + batch_size_, indices_.size());
    const Eigen::Index rows = static_cast<Eigen::Index>(end - start);

    MatrixXf bX(rows, X_.cols());
    MatrixXf bY(rows, Y_.cols());
    for (Eigen::Index i = 0; i < rows; ++i) {
        const auto src = indices_[start + static_cast<std::size_t>(i)];
        bX.row(i) = X_.row(src);
        bY.row(i) = Y_.row(src);
    }
    return {std::move(bX), std::move(bY)};
}

DataLoader::Iterator::Iterator(const DataLoader* loader, std::size_t idx)
    : loader_(loader), idx_(idx) {
}

Batch DataLoader::Iterator::operator*() const {
    return loader_->get_batch(idx_);
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    ++idx_;
    return *this;
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return idx_ != other.idx_;
}

DataLoader::Iterator DataLoader::begin() const {
    return {this, 0};
}

DataLoader::Iterator DataLoader::end() const {
    return {this, num_batches()};
}

}  // namespace nn
