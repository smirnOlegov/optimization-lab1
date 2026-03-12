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

// Плохо обусловленная выпуклая функция: log-sum-exp
// f(x) = log( sum_i exp(a_i^T x - b_i) ) + (mu/2)||x||^2
// Гладкая, строго выпуклая. Аппроксимирует max(Ax - b).
// Плохая обусловленность: экспоненты создают огромный разброс кривизн.
inline ConstructiveReal LogSumExp(const VectorData& x) {
    static const std::vector<std::vector<double>> A = {
        { 3.0, -1.0,  0.5,  0.0,  0.2},
        {-1.0,  4.0, -0.5,  1.0, -0.3},
        { 0.5, -0.5,  5.0, -1.5,  0.7},
        { 0.0,  1.0, -1.5,  3.5, -0.8},
        { 0.2, -0.3,  0.7, -0.8,  6.0},
        {-2.0,  1.5, -1.0,  2.0, -1.0},
    };
    static const std::vector<double> b = {1.0, -0.5, 2.0, -1.0, 0.5, 1.5};

    ConstructiveReal mu(Rational(5, 10)); // 0.5
    ConstructiveReal zero(Rational(0, 1));

    size_t m = A.size();
    size_t n = std::min(x.size(), A[0].size());

    ConstructiveReal sum_exp = zero;

    for (size_t i = 0; i < m; ++i) {
        ConstructiveReal ax_b = zero - ConstructiveReal(Rational(std::round(b[i]*10), 10));

        for (size_t j = 0; j < n; ++j) {
            long long a_ij_val = std::round(A[i][j] * 10);
            ConstructiveReal A_ij(Rational(a_ij_val, 10));
            ax_b = ax_b + (A_ij * x[j]);
        }

        // Используем наш новый метод exp()
        sum_exp = sum_exp + ax_b.exp();
    }

    // Регуляризация (mu / 2) * ||x||^2
    ConstructiveReal norm_sq = zero;
    for (size_t j = 0; j < x.size(); ++j) {
        norm_sq = norm_sq + (x[j] * x[j]);
    }

    ConstructiveReal reg = (mu / ConstructiveReal(Rational(2, 1))) * norm_sq;

    // Используем наш новый метод log()
    return sum_exp.log() + reg;
}
