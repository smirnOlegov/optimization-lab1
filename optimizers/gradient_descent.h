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
    int max_iterations;

public:
    GradientDescentOptimizer(VectorData start, OptimizationType type,
                             double lr = 0.001, int max_iterations = 5000)
        : Optimizer(type), start(std::move(start)), lr(lr), max_iterations(max_iterations) {}

    OptimizationResult optimize(ObjectiveFunction f, size_t /*dim*/,
                                double target_value, double tolerance) override {
        VectorData current = start;
        double sign = (type == OptimizationType::MINIMIZE) ? -1.0 : 1.0;
        double current_value = f(current);

        if (reached_target(current_value, target_value, tolerance)) {
            return {current, current_value, 0, true};
        }

        for (int iteration = 1; iteration <= max_iterations; ++iteration) {
            VectorData grad = numerical_gradient(f, current);
            for (size_t i = 0; i < current.size(); ++i) {
                current[i] += sign * lr * grad[i];
            }
            current_value = f(current);

            if (reached_target(current_value, target_value, tolerance)) {
                return {current, current_value, static_cast<size_t>(iteration), true};
            }
        }

        return {current, current_value, static_cast<size_t>(max_iterations), false};
    }
};
