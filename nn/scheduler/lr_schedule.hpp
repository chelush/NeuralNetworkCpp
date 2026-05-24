#pragma once

#include "core/math/Linalg.h"

namespace nn {

class LRSchedule {
public:
    virtual ~LRSchedule() = default;

    virtual float at(Eigen::Index epoch) const = 0;
};

class ConstantSchedule final : public LRSchedule {
public:
    explicit ConstantSchedule(float lr);

    float at(Eigen::Index epoch) const override;

private:
    float lr_;
};

class StepDecaySchedule final : public LRSchedule {
public:
    StepDecaySchedule(float initial_lr, Eigen::Index step_size, float gamma);

    float at(Eigen::Index epoch) const override;

private:
    float initial_;
    Eigen::Index step_;
    float gamma_;
};

class ExponentialDecaySchedule final : public LRSchedule {
public:
    ExponentialDecaySchedule(float initial_lr, float gamma);

    float at(Eigen::Index epoch) const override;

private:
    float initial_;
    float gamma_;
};

class CosineAnnealingSchedule final : public LRSchedule {
public:
    CosineAnnealingSchedule(float lr_max, float lr_min, Eigen::Index total_epochs);

    float at(Eigen::Index epoch) const override;

private:
    float lr_max_;
    float lr_min_;
    Eigen::Index total_;
};

}  // namespace nn
