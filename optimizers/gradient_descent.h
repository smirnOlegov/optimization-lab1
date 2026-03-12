#pragma once

#include "utils/types.h"
#include "utils/numerical_gradient.h"
#include "optimizer.h"
#include <random>
#include <algorithm>

// Градиентный спуск
class GradientDescentOptimizer : public Optimizer {
    VectorData start;
    double lr;
    int epochs;

public:
    GradientDescentOptimizer(VectorData start, OptimizationType type,
                             double lr = 0.001, int epochs = 5000)
        : Optimizer(type), start(std::move(start)), lr(lr), epochs(epochs) {}

    VectorData optimize(ObjectiveFunction f, size_t /*dim*/) override {
        VectorData current = start;
        double sign = (type == OptimizationType::MINIMIZE) ? -1.0 : 1.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            VectorData grad = numerical_gradient(f, current);
            for (size_t i = 0; i < current.size(); ++i) {
                current[i] += sign * lr * grad[i];
            }
        }
        return current;
    }
};