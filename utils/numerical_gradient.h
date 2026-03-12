#pragma once

#include "types.h"

// Численное дифференцирование методом центральных разностей
inline VectorData numerical_gradient(const ObjectiveFunction& f, const VectorData& x, ConstructiveReal h = ConstructiveReal(Rational(1, 100000))) {
    ConstructiveReal zero_val(Rational(0, 1));
    VectorData grad(x.size(), zero_val);

    for (size_t i = 0; i < x.size(); ++i) {
        VectorData x_plus = x;
        VectorData x_minus = x;

        x_plus[i] = x_plus[i] + h;
        x_minus[i] = x_minus[i] - h;

        grad[i] = (f(x_plus) - f(x_minus)) / (ConstructiveReal(Rational(2, 1)) * h);

        grad[i].collapse();
    }
    return grad;
}