#pragma once

#include "utils/types.h"
#include "optimizer.h"
#include <random>
#include <algorithm>

// Генетический алгоритм (для непрерывных значений)
class GeneticAlgorithmOptimizer : public Optimizer {
    int pop_size;
    int max_generations;

public:
    GeneticAlgorithmOptimizer(OptimizationType type,
                              int pop_size = 100, int max_generations = 500)
        : Optimizer(type), pop_size(pop_size), max_generations(max_generations) {}

    OptimizationResult optimize(ObjectiveFunction f, size_t dim,
                                double target_value, double tolerance) override {
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-5.0, 5.0);
        std::normal_distribution<> mut_dis(0.0, 0.5);

        // Инициализация популяции
        std::vector<VectorData> population(pop_size, VectorData(dim));
        for (int i = 0; i < pop_size; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                population[i][j] = dis(gen);
            }
        }

        auto evaluate = [&](const VectorData& ind) {
            double val = f(ind);
            return (type == OptimizationType::MINIMIZE) ? -val : val;
        };

        auto evaluate_population = [&]() {
            // Оценка фитнеса
            std::vector<std::pair<double, VectorData>> fitness;
            for (const auto& ind : population) {
                fitness.push_back({evaluate(ind), ind});
            }

            // Сортировка по убыванию фитнеса
            std::sort(fitness.begin(), fitness.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            return fitness;
        };

        auto fitness = evaluate_population();
        double best_value = f(fitness.front().second);
        if (reached_target(best_value, target_value, tolerance)) {
            return {fitness.front().second, best_value, 0, true};
        }

        for (int generation = 1; generation <= max_generations; ++generation) {
            std::vector<VectorData> next_gen;
            // Элитизм: сохраняем топ 10%
            int elite_count = pop_size / 10;
            for (int i = 0; i < elite_count; ++i) {
                next_gen.push_back(fitness[i].second);
            }

            // Скрещивание и мутация
            std::uniform_int_distribution<> parent_dis(0, pop_size / 2);
            while (static_cast<int>(next_gen.size()) < pop_size) {
                VectorData p1 = fitness[parent_dis(gen)].second;
                VectorData p2 = fitness[parent_dis(gen)].second;

                VectorData child(dim);
                for (size_t i = 0; i < dim; ++i) {
                    child[i] = (p1[i] + p2[i]) / 2.0;
                    if (std::uniform_real_distribution<>(0, 1)(gen) < 0.2) {
                        child[i] += mut_dis(gen);
                    }
                }
                next_gen.push_back(child);
            }
            population = next_gen;
            fitness = evaluate_population();
            best_value = f(fitness.front().second);

            if (reached_target(best_value, target_value, tolerance)) {
                return {fitness.front().second, best_value,
                        static_cast<size_t>(generation), true};
            }
        }

        return {fitness.front().second, best_value,
                static_cast<size_t>(max_generations), false};
    }
};
