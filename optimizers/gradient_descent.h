#pragma once

#include "../utils/types.h"
#include "../utils/numerical_gradient.h"
#include "optimizer.h"
#include <random>
#include <algorithm>
#include <vector>

// Градиентный спуск на конструктивных числах
class GradientDescentOptimizer : public Optimizer {
    VectorData start;
    ConstructiveReal lr;
    int max_iterations;

    std::vector<double> point_to_double(const VectorData& pt) {
        std::vector<double> res(pt.size());
        for (size_t i = 0; i < pt.size(); ++i) {
            res[i] = pt[i].to_double();
        }
        return res;
    }

public:
    std::vector<std::vector<double>> history;

    GradientDescentOptimizer(VectorData start, OptimizationType type,
                             ConstructiveReal lr = ConstructiveReal(Rational(1, 1000)), // 0.001
                             int max_iterations = 5000)
        : Optimizer(type), start(std::move(start)), lr(lr), max_iterations(max_iterations) {}

    OptimizationResult optimize(ObjectiveFunction f, size_t /*dim*/,
                                ConstructiveReal target_value, ConstructiveReal tolerance) override {
        VectorData current = start;

        // Очищаем историю при каждом новом запуске optimize()
        history.clear();

        // Знак: -1/1 для минимизации, 1/1 для максимизации
        ConstructiveReal sign = (type == OptimizationType::MINIMIZE)
                                ? ConstructiveReal(Rational(-1, 1))
                                : ConstructiveReal(Rational(1, 1));

        ConstructiveReal current_value = f(current);

        // Записываем стартовую точку
        history.push_back(point_to_double(current));

        if (reached_target(current_value, target_value, tolerance)) {
            return {current, current_value, 0, true};
        }

        int collapse_epoch = 10; // Каждые 10 шагов фиксируем состояние
        Rational collapse_precision(1, 1000000000000000LL);

        for (int iteration = 1; iteration <= max_iterations; ++iteration) {
            VectorData grad = numerical_gradient(f, current);

            for (size_t i = 0; i < current.size(); ++i) {
                current[i] = current[i] + (sign * lr * grad[i]);
            }

            current_value = f(current);

            // Периодический коллапс
            if (iteration % collapse_epoch == 0) {
                for (size_t i = 0; i < current.size(); ++i) {
                    current[i].collapse(collapse_precision);
                }
                current_value.collapse(collapse_precision);
            }

            // Записываем текущую точку в историю
            history.push_back(point_to_double(current));

            if (reached_target(current_value, target_value, tolerance)) {
                return {current, current_value, static_cast<size_t>(iteration), true};
            }
        }

        return {current, current_value, static_cast<size_t>(max_iterations), false};
    }
};