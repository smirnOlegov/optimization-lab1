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

// Плохо обусловленная выпуклая функция: log-sum-exp
// f(x) = log( sum_i exp(a_i^T x - b_i) ) + (mu/2)||x||^2
// Гладкая, строго выпуклая. Аппроксимирует max(Ax - b).
// Плохая обусловленность: экспоненты создают огромный разброс кривизн.
// Коэффициенты A и b зашиты — задача фиксированная, воспроизводимая.
inline double LogSumExp(const VectorData& x) {
    // Матрица A (6 строк × dim столбцов) — коэффициенты линейных форм
    // Подобраны с разными масштабами для плохой обусловленности
    static const std::vector<std::vector<double>> A = {
        { 3.0, -1.0,  0.5,  0.0,  0.2},
        {-1.0,  4.0, -0.5,  1.0, -0.3},
        { 0.5, -0.5,  5.0, -1.5,  0.7},
        { 0.0,  1.0, -1.5,  3.5, -0.8},
        { 0.2, -0.3,  0.7, -0.8,  6.0},
        {-2.0,  1.5, -1.0,  2.0, -1.0},
    };
    static const std::vector<double> b = {1.0, -0.5, 2.0, -1.0, 0.5, 1.5};
    static const double mu = 0.5; // регуляризация — гарантирует строгую выпуклость

    size_t m = A.size();
    size_t n = std::min(x.size(), A[0].size());

    // Вычисляем z_i = a_i^T x - b_i
    std::vector<double> z(m);
    double z_max = -1e300;
    for (size_t i = 0; i < m; ++i) {
        z[i] = -b[i];
        for (size_t j = 0; j < n; ++j) {
            z[i] += A[i][j] * x[j];
        }
        if (z[i] > z_max) z_max = z[i];
    }

    // log-sum-exp со сдвигом для численной стабильности: log(sum exp(z_i - z_max)) + z_max
    double sum_exp = 0.0;
    for (size_t i = 0; i < m; ++i) {
        sum_exp += std::exp(z[i] - z_max);
    }

    double lse = z_max + std::log(sum_exp);

    // Регуляризация: (mu/2) * ||x||^2
    double norm_sq = 0.0;
    for (size_t j = 0; j < x.size(); ++j) {
        norm_sq += x[j] * x[j];
    }

    return lse + 0.5 * mu * norm_sq;
}
