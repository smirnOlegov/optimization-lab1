#pragma once

#include "../utils/types.h"
#include <cmath>
#include <random>

struct OptimizationResult {
    VectorData point;
    ConstructiveReal value;
    size_t iterations;
    bool converged;
};

// Базовый класс оптимизатора
class Optimizer {
protected:
    OptimizationType type;

    static bool reached_target(const ConstructiveReal &value, const ConstructiveReal& target_value, const ConstructiveReal &tolerance) {
        return (value - target_value < tolerance) && (target_value - value < tolerance);
    }

public:
    std::vector<VectorData> history;

    Optimizer(OptimizationType type) : type(type) {}
    virtual ~Optimizer() = default;

    virtual OptimizationResult optimize(ObjectiveFunction f, size_t dim,
                                        ConstructiveReal target_value, ConstructiveReal tolerance) = 0;
};

