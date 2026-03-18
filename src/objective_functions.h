#pragma once

#include "../utils/types.h"
#include <cmath>

// Функция Розенброка 2D: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
inline ConstructiveReal Rosenbrock2D(const VectorData& x) {
    ConstructiveReal zero(Rational(0, 1));
    if (x.size() < 2) return zero;

    ConstructiveReal one(Rational(1, 1));
    ConstructiveReal hundred(Rational(100, 1));

    ConstructiveReal t1 = one - x[0];
    ConstructiveReal t2 = x[1] - (x[0] * x[0]);

    return (t1 * t1) + (hundred * (t2 * t2));
}

// Функция Розенброка N-мерная
inline ConstructiveReal RosenbrockND(const VectorData& x) {
    ConstructiveReal sum(Rational(0, 1));
    ConstructiveReal one(Rational(1, 1));
    ConstructiveReal hundred(Rational(100, 1));

    for (size_t i = 0; i < x.size() - 1; ++i) {
        ConstructiveReal t1 = x[i+1] - (x[i] * x[i]);
        ConstructiveReal t2 = one - x[i];
        sum = sum + (hundred * (t1 * t1)) + (t2 * t2);
    }
    return sum;
}