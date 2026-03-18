#pragma once

#include "../utils/types.h"
#include "optimizer.h"
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

class GeneticAlgorithmOptimizer : public Optimizer {
    int pop_size;
    int max_generations;

    static ConstructiveReal double_to_creal(double val) {
        long long num = std::round(val * 1000000.0);
        return {Rational(num, 1000000)};
    }

    std::vector<double> point_to_double(const VectorData& pt) {
        std::vector<double> res(pt.size());
        for (size_t i = 0; i < pt.size(); ++i) {
            res[i] = pt[i].to_double();
        }
        return res;
    }

public:
    std::vector<std::vector<double>> history;

    GeneticAlgorithmOptimizer(OptimizationType type,
                              int pop_size = 500, int max_generations = 10000)
        : Optimizer(type), pop_size(pop_size), max_generations(max_generations) {}

    OptimizationResult optimize(ObjectiveFunction f, size_t dim,
                                ConstructiveReal target_value, ConstructiveReal tolerance) override {
        history.clear();

        std::mt19937 gen(42);
        std::uniform_real_distribution<> start_dis(-2.0, 2.0);
        std::normal_distribution<> mut_dis(0.0, 0.02);
        std::uniform_real_distribution<> alpha_dis(-0.1, 1.1);

        ConstructiveReal zero_val(Rational(0, 1));
        ConstructiveReal one_val(Rational(1, 1));

        std::vector<VectorData> population(pop_size, VectorData(dim, zero_val));
        for (int i = 0; i < pop_size; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                population[i][j] = double_to_creal(start_dis(gen));
            }
        }

        Rational ga_precision(1, 1000000000LL);
        auto evaluate = [&](const VectorData& ind) {
            ConstructiveReal val = f(ind);
            val.collapse(ga_precision);
            if (type == OptimizationType::MINIMIZE) return zero_val - val;
            return val;
        };

        auto evaluate_population = [&]() {
            std::vector<std::pair<ConstructiveReal, VectorData>> fitness;
            fitness.reserve(pop_size);
            for (const auto& ind : population) {
                fitness.push_back({evaluate(ind), ind});
            }
            std::sort(fitness.begin(), fitness.end(), [](const auto& a, const auto& b) {
                return b.first < a.first;
            });
            return fitness;
        };

        auto fitness = evaluate_population();
        ConstructiveReal best_value = f(fitness.front().second);

        // Записываем лучшую особь нулевого поколения в историю
        history.push_back(point_to_double(fitness.front().second));

        if (reached_target(best_value, target_value, tolerance)) {
            return {fitness.front().second, best_value, 0, true};
        }

        auto tournament_select = [&]() {
            std::uniform_int_distribution<> t_dis(0, pop_size - 1);
            int best_idx = t_dis(gen);
            for (int i = 0; i < 3; ++i) {
                int idx = t_dis(gen);
                if (idx < best_idx) best_idx = idx;
            }
            return fitness[best_idx].second;
        };

        int collapse_epoch = 5;
        Rational collapse_precision(1, 1000000000000LL);

        for (int generation = 1; generation <= max_generations; ++generation) {
            std::vector<VectorData> next_gen;
            next_gen.reserve(pop_size);

            // Элитизм: сохраняем топ-10% в неприкосновенности
            int elite_count = pop_size / 10;
            for (int i = 0; i < elite_count; ++i) {
                next_gen.push_back(fitness[i].second);
            }

            // Заполняем остаток популяции
            while (static_cast<int>(next_gen.size()) < pop_size) {
                VectorData p1 = tournament_select();
                VectorData p2 = tournament_select();

                VectorData child(dim, zero_val);

                for (size_t i = 0; i < dim; ++i) {
                    ConstructiveReal alpha = double_to_creal(alpha_dis(gen));
                    child[i] = (alpha * p1[i]) + ((one_val - alpha) * p2[i]);

                    if (std::uniform_real_distribution<>(0, 1)(gen) < (1.0 / dim)) {
                        child[i] = child[i] + double_to_creal(mut_dis(gen));
                    }

                    child[i].collapse(ga_precision);
                }
                next_gen.push_back(child);
            }

            population = next_gen;

            if (generation % collapse_epoch == 0) {
                for (auto& ind : population) {
                    for (auto& gene : ind) {
                        gene.collapse(collapse_precision);
                    }
                }
            }

            fitness = evaluate_population();
            best_value = f(fitness.front().second);
            best_value.collapse(ga_precision);

            if (generation % collapse_epoch == 0) {
                best_value.collapse(collapse_precision);
            }

            // Сохраняем лучшую особь текущего поколения
            history.push_back(point_to_double(fitness.front().second));

            if (reached_target(best_value, target_value, tolerance)) {
                return {fitness.front().second, best_value,
                        static_cast<size_t>(generation), true};
            }
        }


        return {fitness.front().second, best_value,
                static_cast<size_t>(max_generations), false};
    }
};