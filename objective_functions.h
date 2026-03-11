#pragma once

#include "types.h"
#include <cmath>

// Функция Розенброка 2D: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
double Rosenbrock2D(const VectorData& x) {
    if (x.size() < 2) return 0.0;
    return std::pow(1.0 - x[0], 2) + 100.0 * std::pow(x[1] - std::pow(x[0], 2), 2);
}

// Функция Розенброка N-мерная
double RosenbrockND(const VectorData& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; ++i) {
        sum += 100.0 * std::pow(x[i+1] - std::pow(x[i], 2), 2) + std::pow(1.0 - x[i], 2);
    }
    return sum;
}
