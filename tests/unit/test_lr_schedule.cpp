#include "nn/scheduler/lr_schedule.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

TEST(ConstantScheduleTest, AlwaysReturnsSameValue) {
    nn::ConstantSchedule sch(0.05f);
    EXPECT_FLOAT_EQ(sch.at(0), 0.05f);
    EXPECT_FLOAT_EQ(sch.at(7), 0.05f);
    EXPECT_FLOAT_EQ(sch.at(1000), 0.05f);
}

TEST(ConstantScheduleTest, RejectsNonPositiveLr) {
    EXPECT_THROW(nn::ConstantSchedule(0.0f), std::invalid_argument);
    EXPECT_THROW(nn::ConstantSchedule(-1.0f), std::invalid_argument);
}

TEST(StepDecayScheduleTest, MatchesFormula) {
    nn::StepDecaySchedule sch(0.1f, 3, 0.5f);
    // Epoch 0..2 -> 0.1, 3..5 -> 0.05, 6..8 -> 0.025
    EXPECT_FLOAT_EQ(sch.at(0), 0.1f);
    EXPECT_FLOAT_EQ(sch.at(2), 0.1f);
    EXPECT_FLOAT_EQ(sch.at(3), 0.05f);
    EXPECT_FLOAT_EQ(sch.at(5), 0.05f);
    EXPECT_FLOAT_EQ(sch.at(6), 0.025f);
}

TEST(StepDecayScheduleTest, RejectsBadParameters) {
    EXPECT_THROW(nn::StepDecaySchedule(0.0f, 1, 0.5f), std::invalid_argument);
    EXPECT_THROW(nn::StepDecaySchedule(0.1f, 0, 0.5f), std::invalid_argument);
    EXPECT_THROW(nn::StepDecaySchedule(0.1f, 1, 0.0f), std::invalid_argument);
    EXPECT_THROW(nn::StepDecaySchedule(0.1f, 1, 1.5f), std::invalid_argument);
}

TEST(ExponentialDecayScheduleTest, MatchesFormula) {
    nn::ExponentialDecaySchedule sch(0.1f, 0.9f);
    EXPECT_FLOAT_EQ(sch.at(0), 0.1f);
    EXPECT_NEAR(sch.at(1), 0.1f * 0.9f, 1e-6f);
    EXPECT_NEAR(sch.at(3), 0.1f * std::pow(0.9f, 3.0f), 1e-6f);
}

TEST(ExponentialDecayScheduleTest, RejectsBadParameters) {
    EXPECT_THROW(nn::ExponentialDecaySchedule(-0.1f, 0.9f), std::invalid_argument);
    EXPECT_THROW(nn::ExponentialDecaySchedule(0.1f, 1.5f), std::invalid_argument);
    EXPECT_THROW(nn::ExponentialDecaySchedule(0.1f, 0.0f), std::invalid_argument);
}

TEST(CosineAnnealingScheduleTest, BoundsAndMonotonicity) {
    nn::CosineAnnealingSchedule sch(0.1f, 0.001f, 10);

    EXPECT_NEAR(sch.at(0), 0.1f, 1e-6f);
    EXPECT_NEAR(sch.at(10), 0.001f, 1e-6f);

    float prev = sch.at(0);
    for (Eigen::Index e = 1; e <= 10; ++e) {
        const float cur = sch.at(e);
        EXPECT_LE(cur, prev + 1e-6f);
        EXPECT_GE(cur, 0.001f - 1e-6f);
        prev = cur;
    }

    // Out-of-range clamps to lr_min.
    EXPECT_NEAR(sch.at(100), 0.001f, 1e-6f);
}

TEST(CosineAnnealingScheduleTest, MidpointMatchesFormula) {
    nn::CosineAnnealingSchedule sch(1.0f, 0.0001f, 10);
    // At epoch = total / 2 cos(pi/2) = 0, => lr_min + 0.5*(lr_max - lr_min)
    const float expected = 0.0001f + 0.5f * (1.0f - 0.0001f);
    EXPECT_NEAR(sch.at(5), expected, 1e-4f);
}

TEST(CosineAnnealingScheduleTest, RejectsBadParameters) {
    EXPECT_THROW(nn::CosineAnnealingSchedule(0.0f, 0.0f, 10), std::invalid_argument);
    EXPECT_THROW(nn::CosineAnnealingSchedule(0.1f, 0.0f, 10), std::invalid_argument);
    EXPECT_THROW(nn::CosineAnnealingSchedule(0.1f, 0.2f, 10), std::invalid_argument);
    EXPECT_THROW(nn::CosineAnnealingSchedule(0.1f, 0.05f, 0), std::invalid_argument);
}
