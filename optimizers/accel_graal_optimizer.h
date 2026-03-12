#pragma once

#include "optimizer.h"
#include "../utils/numerical_gradient.h"
#include "../utils/vector_ops.h"
#include "../src/constructive_real.h"
#include <algorithm>

// Accelerated GRAAL optimizer from "Nesterov Finds GRAAL" (arXiv:2507.09823)
class AcceleratedGRAALOptimizer : public Optimizer {
    VectorData start;
    int max_iterations;
    ConstructiveReal eta_0;
    ConstructiveReal theta;
    ConstructiveReal gamma;
    ConstructiveReal nu;

    // Вспомогательные константы
    static ConstructiveReal zero() { return {Rational(0, 1)}; }
    static ConstructiveReal one() { return {Rational(1, 1)}; }
    static ConstructiveReal two() { return {Rational(2, 1)}; }
    static ConstructiveReal four() { return {Rational(4, 1)}; }
    static ConstructiveReal tiny_eps() { return {Rational(1, 1000000)}; } // 1e-6 вместо 1e-9 вместо 1e-30 ;)
    static ConstructiveReal huge_val() { return {Rational(10000, 1)}; } // вместо infinity 10^4

    // Bregman divergence: B_f(x, z) = f(z) - f(x) - <grad_x, z - x>
    static ConstructiveReal bregman_divergence(const ObjectiveFunction& f,
                                               const VectorData& x, const VectorData& z,
                                               const VectorData& grad_x) {
        return f(z) - f(x) - vec_dot(grad_x, vec_sub(z, x));
    }

    // Curvature estimator: Lambda(x, z) = 2*B_f(x,z) / ||grad_x - grad_z||^2
    static ConstructiveReal curvature_estimator(const ObjectiveFunction& f,
                                                const VectorData& x, const VectorData& z,
                                                const VectorData& grad_x, const VectorData& grad_z) {
        ConstructiveReal grad_diff_sq = vec_norm_sq(vec_sub(grad_x, grad_z));
        if (grad_diff_sq < tiny_eps()) return huge_val();
        ConstructiveReal bf = bregman_divergence(f, x, z, grad_x);
        return two() * bf / grad_diff_sq;
    }

    // Безопасный минимум для ConstructiveReal
    static ConstructiveReal min_cr(const ConstructiveReal& a, const ConstructiveReal& b) {
        return (a < b) ? a : b;
    }

    // Безопасный clamp для ConstructiveReal
    static ConstructiveReal clamp_cr(const ConstructiveReal& v, const ConstructiveReal& lo, const ConstructiveReal& hi) {
        if (v < lo) return lo;
        if (hi < v) return hi;
        return v;
    }

public:
    AcceleratedGRAALOptimizer(VectorData start, OptimizationType type,
                              int max_iterations = 10000,
                              const ConstructiveReal &eta_0 = ConstructiveReal(Rational(1, 100)),   // 0.01
                              const ConstructiveReal &theta = ConstructiveReal(Rational(10, 1)),    // 10.0
                              const ConstructiveReal &gamma = ConstructiveReal(Rational(1, 100)))   // 0.01
        : Optimizer(type), start(std::move(start)),
          max_iterations(max_iterations), eta_0(eta_0),
          theta(theta), gamma(gamma),
          nu(Rational(0, 1)) // инициализация пустышкой
    {
        // nu = gamma / (4 * theta * (1+gamma)^2)  — from constraint (19)
        ConstructiveReal one_plus_gamma = one() + gamma;
        nu = gamma / (four() * theta * one_plus_gamma * one_plus_gamma);
    }

    OptimizationResult optimize(ObjectiveFunction f_orig, size_t /*dim*/,
                                ConstructiveReal target_value, ConstructiveReal tolerance) override {
        // For MAXIMIZE: negate objective so we always minimize
        ObjectiveFunction f = f_orig;
        if (type == OptimizationType::MAXIMIZE) {
            f = [&f_orig, this](const VectorData& x) { return zero() - f_orig(x); };
        }

        // Initialize: x_0 = x_tilde_0 = x_bar_0 = start
        VectorData x_k = start;
        VectorData x_tilde_k = start;
        VectorData x_bar_k = start;

        ConstructiveReal eta_k = eta_0;
        ConstructiveReal eta_km1 = eta_0;   // eta_{k-1}
        ConstructiveReal H_k = eta_0;
        ConstructiveReal H_km1 = zero();      // H_{k-1}

        VectorData grad_tilde_k = numerical_gradient(f, x_tilde_k);

        // Track best point seen (practical for non-convex)
        VectorData best_x = start;
        ConstructiveReal best_transformed_value = f(start);
        ConstructiveReal best_original_value = f_orig(start);

        if (reached_target(best_original_value, target_value, tolerance)) {
            return {best_x, best_original_value, 0, true};
        }

        auto consider_candidate = [&](const VectorData& point,
                                      const ConstructiveReal &transformed_value,
                                      const ConstructiveReal &original_value) {
            if (transformed_value < best_transformed_value) {
                best_transformed_value = transformed_value;
                best_original_value = original_value;
                best_x = point;
            }
        };

        ConstructiveReal one_plus_gamma = one() + gamma;

        // Лимиты для clamp: 1e-15 до 1e6
        ConstructiveReal eta_min(Rational(1, 10000000)); // Не можем использовать 1e-15, возьмем 1e-7
        ConstructiveReal eta_max(Rational(1000, 1));    // 1e3

        for (int iteration = 1; iteration <= max_iterations; ++iteration) {
            // Step 1: alpha_{k+1}
            ConstructiveReal alpha_kp1 = (one_plus_gamma * eta_k) / (H_k + one_plus_gamma * eta_k);

            // Step 2: x_{k+1} = x_k - eta_k * grad_f(x_tilde_k)
            VectorData x_kp1 = vec_sub(x_k, vec_scale(eta_k, grad_tilde_k));

            // Step 3: x_bar_{k+1} = beta_k * x_tilde_k + (1 - beta_k) * x_bar_k
            ConstructiveReal beta_k = eta_k / (alpha_kp1 * (H_k + one_plus_gamma * eta_k));
            VectorData x_bar_kp1 = vec_linear_combo(beta_k, x_tilde_k,
                                                     one() - beta_k, x_bar_k);

            // Step 4: x_hat_{k+1} = x_{k+1} + theta * (x_{k+1} - x_k)  — GRAAL extrapolation
            VectorData x_hat_kp1 = vec_add(x_kp1, vec_scale(theta, vec_sub(x_kp1, x_k)));

            // Step 5: x_tilde_{k+1} = alpha_{k+1} * x_hat_{k+1} + (1 - alpha_{k+1}) * x_bar_{k+1}
            VectorData x_tilde_kp1 = vec_linear_combo(alpha_kp1, x_hat_kp1,
                                                       one() - alpha_kp1, x_bar_kp1);

            // Compute gradient at new x_tilde
            VectorData grad_tilde_kp1 = numerical_gradient(f, x_tilde_kp1);

            // Step 6: curvature estimates
            VectorData grad_bar_kp1 = numerical_gradient(f, x_bar_kp1);
            ConstructiveReal lambda_1 = curvature_estimator(f, x_bar_kp1, x_tilde_k,
                                                   grad_bar_kp1, grad_tilde_k);
            ConstructiveReal lambda_2 = curvature_estimator(f, x_bar_kp1, x_tilde_kp1,
                                                   grad_bar_kp1, grad_tilde_kp1);
            ConstructiveReal lambda_kp1 = min_cr(lambda_1, lambda_2);

            // Step 7: adaptive stepsize
            // When lambda <= 0 (non-convex region), curvature estimate is unreliable — use growing step
            ConstructiveReal eta_grow = one_plus_gamma * eta_k;

            ConstructiveReal eta_adapt = eta_grow;
            if ((zero() < H_km1) && (zero() < eta_km1) && (zero() < lambda_kp1)) {
                 eta_adapt = nu * H_km1 * lambda_kp1 / eta_km1;
            }

            ConstructiveReal eta_kp1 = min_cr(eta_grow, eta_adapt);

            // Numerical safety: clamp eta
            eta_kp1 = clamp_cr(eta_kp1, eta_min, eta_max);

            // Step 8: H_{k+1}
            ConstructiveReal H_kp1 = H_k + eta_kp1;

            // Track best iterate among practical return candidates.
            ConstructiveReal x_kp1_value = f(x_kp1);
            ConstructiveReal x_kp1_original_value = f_orig(x_kp1);
            consider_candidate(x_kp1, x_kp1_value, x_kp1_original_value);

            ConstructiveReal x_bar_kp1_value = f(x_bar_kp1);
            ConstructiveReal x_bar_kp1_original_value = f_orig(x_bar_kp1);
            consider_candidate(x_bar_kp1, x_bar_kp1_value, x_bar_kp1_original_value);

            if (reached_target(best_original_value, target_value, tolerance)) {
                return {best_x, best_original_value,
                        static_cast<size_t>(iteration), true};
            }

            // === ДОБАВИТЬ ЭТОТ БЛОК: СХЛОПЫВАНИЕ ДЕРЕВЬЕВ ===
            // Задаем точность схлопывания (например, 1e-7).
            // Это обрубит деревья и превратит их в ConstNode.
            eta_kp1.collapse();
            H_kp1.collapse();

            for (size_t i = 0; i < x_kp1.size(); ++i) {
                x_kp1[i].collapse();
                x_bar_kp1[i].collapse();
                x_tilde_kp1[i].collapse();
                grad_tilde_kp1[i].collapse();
            }

            // Shift variables for next iteration
            H_km1 = H_k;
            H_k = H_kp1;
            eta_km1 = eta_k;
            eta_k = eta_kp1;
            x_k = x_kp1;
            x_bar_k = x_bar_kp1;
            x_tilde_k = x_tilde_kp1;
            grad_tilde_k = grad_tilde_kp1;
        }

        return {best_x, best_original_value,
                static_cast<size_t>(max_iterations), false};
    }
};