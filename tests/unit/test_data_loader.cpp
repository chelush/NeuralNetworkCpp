#include "utils/data_loader.hpp"
#include <gtest/gtest.h>
#include <random>
#include <set>
#include <stdexcept>

TEST(DataLoaderTest, NumBatchesRoundsUp) {
    nn::MatrixXf X = nn::MatrixXf::Random(10, 3);
    nn::MatrixXf Y = nn::MatrixXf::Random(10, 1);
    nn::DataLoader loader(std::move(X), std::move(Y), nn::BatchSize{3}, nn::Shuffle{false});

    EXPECT_EQ(loader.size(), 10u);
    EXPECT_EQ(loader.batch_size(), 3u);
    EXPECT_EQ(loader.num_batches(), 4u);
}

TEST(DataLoaderTest, NoShuffleIteratesInOrderAndCoversEverything) {
    nn::MatrixXf X(5, 1);
    X << 0.0f, 1.0f, 2.0f, 3.0f, 4.0f;
    nn::MatrixXf Y(5, 1);
    Y << 10.0f, 11.0f, 12.0f, 13.0f, 14.0f;

    nn::DataLoader loader(std::move(X), std::move(Y), nn::BatchSize{2}, nn::Shuffle{false});

    std::vector<float> seen_x;
    std::vector<float> seen_y;
    for (auto batch : loader) {
        for (Eigen::Index i = 0; i < batch.X.rows(); ++i) {
            seen_x.push_back(batch.X(i, 0));
            seen_y.push_back(batch.Y(i, 0));
        }
    }

    ASSERT_EQ(seen_x.size(), 5u);
    EXPECT_EQ(seen_x[0], 0.0f);
    EXPECT_EQ(seen_x[1], 1.0f);
    EXPECT_EQ(seen_x[4], 4.0f);
    EXPECT_EQ(seen_y[0], 10.0f);
    EXPECT_EQ(seen_y[4], 14.0f);
}

TEST(DataLoaderTest, ShuffleCoversAllSamplesOnce) {
    nn::MatrixXf X(20, 1);
    for (int i = 0; i < 20; ++i)
        X(i, 0) = static_cast<float>(i);
    nn::MatrixXf Y = X;

    nn::DataLoader loader(std::move(X), std::move(Y), nn::BatchSize{4}, nn::Shuffle{true});

    std::mt19937 engine(123);
    loader.reset_epoch(engine);

    std::set<int> seen;
    for (auto batch : loader) {
        for (Eigen::Index i = 0; i < batch.X.rows(); ++i) {
            seen.insert(static_cast<int>(batch.X(i, 0)));
            EXPECT_EQ(batch.X(i, 0), batch.Y(i, 0));
        }
    }
    EXPECT_EQ(seen.size(), 20u);
}

TEST(DataLoaderTest, RejectsBadInputs) {
    EXPECT_THROW(nn::DataLoader(nn::MatrixXf::Zero(3, 2), nn::MatrixXf::Zero(4, 1),
                                nn::BatchSize{2}, nn::Shuffle{false}),
                 std::invalid_argument);
    EXPECT_THROW(nn::DataLoader(nn::MatrixXf::Zero(3, 2), nn::MatrixXf::Zero(3, 1),
                                nn::BatchSize{0}, nn::Shuffle{false}),
                 std::invalid_argument);
}

TEST(DataLoaderTest, GetBatchOutOfRangeThrows) {
    nn::DataLoader loader(nn::MatrixXf::Zero(4, 1), nn::MatrixXf::Zero(4, 1), nn::BatchSize{2},
                          nn::Shuffle{false});
    EXPECT_NO_THROW(loader.get_batch(0));
    EXPECT_NO_THROW(loader.get_batch(1));
    EXPECT_THROW(loader.get_batch(2), std::invalid_argument);
}
