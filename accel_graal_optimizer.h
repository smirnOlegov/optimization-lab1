#pragma once

#include "optimizer.h"
#include "vector_ops.h"
#include <cmath>
#include <limits>
#include <algorithm>

// Accelerated GRAAL optimizer from "Nesterov Finds GRAAL" (arXiv:2507.09823)
class AcceleratedGRAALOptimizer : public Optimizer {
    VectorData start;
    int max_iterations;
    double eta_0;
    double theta;
    double gamma;
    double nu;

    // Bregman divergence: B_f(x, z) = f(z) - f(x) - <grad_x, z - x>
    double bregman_divergence(const ObjectiveFunction& f,
                              const VectorData& x, const VectorData& z,
                              const VectorData& grad_x) {
        return f(z) - f(x) - vec_dot(grad_x, vec_sub(z, x));
    }

    // Curvature estimator: Lambda(x, z) = 2*B_f(x,z) / ||grad_x - grad_z||^2
    double curvature_estimator(const ObjectiveFunction& f,
                               const VectorData& x, const VectorData& z,
                               const VectorData& grad_x, const VectorData& grad_z) {
        double grad_diff_sq = vec_norm_sq(vec_sub(grad_x, grad_z));
        if (grad_diff_sq < 1e-30) return std::numeric_limits<double>::infinity();
        double bf = bregman_divergence(f, x, z, grad_x);
        return 2.0 * bf / grad_diff_sq;
    }

public:
    AcceleratedGRAALOptimizer(VectorData start, OptimizationType type,
                              int max_iterations = 10000,
                              double eta_0 = 0.01,
                              double theta = 10.0, double gamma = 0.01)
        : Optimizer(type), start(std::move(start)),
          max_iterations(max_iterations), eta_0(eta_0),
          theta(theta), gamma(gamma) {
        // nu = gamma / (4 * theta * (1+gamma)^2)  — from constraint (19)
        double one_plus_gamma = 1.0 + gamma;
        nu = gamma / (4.0 * theta * one_plus_gamma * one_plus_gamma);
    }

    VectorData optimize(ObjectiveFunction f_orig, size_t /*dim*/) override {
        // For MAXIMIZE: negate objective so we always minimize
        ObjectiveFunction f = f_orig;
        if (type == OptimizationType::MAXIMIZE) {
            f = [&f_orig](const VectorData& x) { return -f_orig(x); };
        }

        size_t n = start.size();

        // Initialize: x_0 = x_tilde_0 = x_bar_0 = start
        VectorData x_k = start;
        VectorData x_tilde_k = start;
        VectorData x_bar_k = start;

        double eta_k = eta_0;
        double eta_km1 = eta_0;   // eta_{k-1}
        double H_k = eta_0;
        double H_km1 = 0.0;      // H_{k-1}

        VectorData grad_tilde_k = numerical_gradient(f, x_tilde_k);

        // Track best point seen (practical for non-convex)
        VectorData best_x = start;
        double best_val = f(start);

        for (int k = 0; k < max_iterations; ++k) {
            // Step 1: alpha_{k+1}
            double alpha_kp1 = (1.0 + gamma) * eta_k / (H_k + (1.0 + gamma) * eta_k);

            // Step 2: x_{k+1} = x_k - eta_k * grad_f(x_tilde_k)
            VectorData x_kp1 = vec_sub(x_k, vec_scale(eta_k, grad_tilde_k));

            // Step 3: x_bar_{k+1} = beta_k * x_tilde_k + (1 - beta_k) * x_bar_k
            double beta_k = eta_k / (alpha_kp1 * (H_k + (1.0 + gamma) * eta_k));
            VectorData x_bar_kp1 = vec_linear_combo(beta_k, x_tilde_k,
                                                     1.0 - beta_k, x_bar_k);

            // Step 4: x_hat_{k+1} = x_{k+1} + theta * (x_{k+1} - x_k)  — GRAAL extrapolation
            VectorData x_hat_kp1 = vec_add(x_kp1, vec_scale(theta, vec_sub(x_kp1, x_k)));

            // Step 5: x_tilde_{k+1} = alpha_{k+1} * x_hat_{k+1} + (1 - alpha_{k+1}) * x_bar_{k+1}
            VectorData x_tilde_kp1 = vec_linear_combo(alpha_kp1, x_hat_kp1,
                                                       1.0 - alpha_kp1, x_bar_kp1);

            // Compute gradient at new x_tilde
            VectorData grad_tilde_kp1 = numerical_gradient(f, x_tilde_kp1);

            // Step 6: curvature estimates
            VectorData grad_bar_kp1 = numerical_gradient(f, x_bar_kp1);
            double lambda_1 = curvature_estimator(f, x_bar_kp1, x_tilde_k,
                                                   grad_bar_kp1, grad_tilde_k);
            double lambda_2 = curvature_estimator(f, x_bar_kp1, x_tilde_kp1,
                                                   grad_bar_kp1, grad_tilde_kp1);
            double lambda_kp1 = std::min(lambda_1, lambda_2);

            // Step 7: adaptive stepsize
            // When lambda <= 0 (non-convex region), curvature estimate is unreliable — use growing step
            double eta_grow = (1.0 + gamma) * eta_k;
            double eta_adapt = (H_km1 > 0.0 && eta_km1 > 0.0
                                && std::isfinite(lambda_kp1) && lambda_kp1 > 0.0)
                ? nu * H_km1 * lambda_kp1 / eta_km1
                : eta_grow;
            double eta_kp1 = std::min(eta_grow, eta_adapt);

            // Numerical safety: clamp eta
            eta_kp1 = std::clamp(eta_kp1, 1e-15, 1e6);

            // Step 8: H_{k+1}
            double H_kp1 = H_k + eta_kp1;

            // Track best iterate
            double val_kp1 = f(x_kp1);
            if (val_kp1 < best_val) {
                best_val = val_kp1;
                best_x = x_kp1;
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

        // Return best point seen (practical choice for non-convex objectives)
        // For convex objectives, x_bar_k (ergodic average) has theoretical guarantees
        double bar_val = f(x_bar_k);
        return (bar_val < best_val) ? x_bar_k : best_x;
    }
};
