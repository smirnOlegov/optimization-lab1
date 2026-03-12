#pragma once

#include "utils/types.h"
#include <random>

// Базовый класс оптимизатора
class Optimizer {
protected:
    OptimizationType type;

public:
    Optimizer(OptimizationType type) : type(type) {}
    virtual ~Optimizer() = default;

    virtual VectorData optimize(ObjectiveFunction f, size_t dim) = 0;
};


