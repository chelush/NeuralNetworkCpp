#include "nn/scheduler/lr_schedule.hpp"
#include "app/except.h"
#include <algorithm>
#include <cmath>

namespace nn {

namespace {
constexpr float kPi = 3.14159265358979323846f;
}  // namespace

ConstantSchedule::ConstantSchedule(float lr) : lr_(lr) {
    NN_REQUIRE(lr_ > 0.0f, "ConstantSchedule: lr must be > 0");
}

float ConstantSchedule::at(Eigen::Index) const {
    return lr_;
}

StepDecaySchedule::StepDecaySchedule(float initial_lr, Eigen::Index step_size, float gamma)
    : initial_(initial_lr), step_(step_size), gamma_(gamma) {
    NN_REQUIRE(initial_ > 0.0f, "StepDecaySchedule: initial lr must be > 0");
    NN_REQUIRE(step_ > 0, "StepDecaySchedule: step_size must be > 0");
    NN_REQUIRE(gamma_ > 0.0f && gamma_ <= 1.0f, "StepDecaySchedule: gamma must be in (0, 1]");
}

float StepDecaySchedule::at(Eigen::Index epoch) const {
    NN_REQUIRE(epoch >= 0, "StepDecaySchedule::at(): epoch must be >= 0");
    const Eigen::Index power = epoch / step_;
    return initial_ * std::pow(gamma_, static_cast<float>(power));
}

ExponentialDecaySchedule::ExponentialDecaySchedule(float initial_lr, float gamma)
    : initial_(initial_lr), gamma_(gamma) {
    NN_REQUIRE(initial_ > 0.0f, "ExponentialDecaySchedule: initial lr must be > 0");
    NN_REQUIRE(gamma_ > 0.0f && gamma_ <= 1.0f,
               "ExponentialDecaySchedule: gamma must be in (0, 1]");
}

float ExponentialDecaySchedule::at(Eigen::Index epoch) const {
    NN_REQUIRE(epoch >= 0, "ExponentialDecaySchedule::at(): epoch must be >= 0");
    return initial_ * std::pow(gamma_, static_cast<float>(epoch));
}

CosineAnnealingSchedule::CosineAnnealingSchedule(float lr_max, float lr_min,
                                                 Eigen::Index total_epochs)
    : lr_max_(lr_max), lr_min_(lr_min), total_(total_epochs) {
    NN_REQUIRE(lr_max_ > 0.0f, "CosineAnnealingSchedule: lr_max must be > 0");
    NN_REQUIRE(lr_min_ > 0.0f, "CosineAnnealingSchedule: lr_min must be > 0");
    NN_REQUIRE(lr_min_ <= lr_max_, "CosineAnnealingSchedule: lr_min must be <= lr_max");
    NN_REQUIRE(total_ > 0, "CosineAnnealingSchedule: total_epochs must be > 0");
}

float CosineAnnealingSchedule::at(Eigen::Index epoch) const {
    NN_REQUIRE(epoch >= 0, "CosineAnnealingSchedule::at(): epoch must be >= 0");
    const Eigen::Index clamped = std::min(epoch, total_);
    const float ratio = static_cast<float>(clamped) / static_cast<float>(total_);
    const float cos_term = 0.5f * (1.0f + std::cos(kPi * ratio));
    return lr_min_ + (lr_max_ - lr_min_) * cos_term;
}

}  // namespace nn
