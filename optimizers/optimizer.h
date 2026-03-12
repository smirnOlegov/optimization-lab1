#pragma once

#include "utils/types.h"
#include <cmath>
#include <random>

struct OptimizationResult {
    VectorData point;
    double value;
    size_t iterations;
    bool converged;
};

// Базовый класс оптимизатора
class Optimizer {
protected:
    OptimizationType type;

    bool reached_target(double value, double target_value, double tolerance) const {
        return std::abs(value - target_value) <= tolerance;
    }

public:
    Optimizer(OptimizationType type) : type(type) {}
    virtual ~Optimizer() = default;

    virtual OptimizationResult optimize(ObjectiveFunction f, size_t dim,
                                        double target_value, double tolerance) = 0;
};

