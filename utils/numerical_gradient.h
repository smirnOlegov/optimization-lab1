#pragma once

#include "types.h"

// Численное дифференцирование методом центральных разностей
VectorData numerical_gradient(const ObjectiveFunction& f, const VectorData& x, double h = 1e-5) {
    VectorData grad(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        VectorData x_plus = x;
        VectorData x_minus = x;

        x_plus[i] += h;
        x_minus[i] -= h;

        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * h);
    }
    return grad;
}
