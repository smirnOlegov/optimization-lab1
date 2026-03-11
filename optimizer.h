#pragma once

#include "types.h"
#include "numerical_gradient.h"
#include <random>
#include <algorithm>

// Базовый класс оптимизатора
class Optimizer {
protected:
    OptimizationType type;

public:
    Optimizer(OptimizationType type) : type(type) {}
    virtual ~Optimizer() = default;

    virtual VectorData optimize(ObjectiveFunction f, size_t dim) = 0;
};

// Градиентный спуск
class GradientDescentOptimizer : public Optimizer {
    VectorData start;
    double lr;
    int epochs;

public:
    GradientDescentOptimizer(VectorData start, OptimizationType type,
                             double lr = 0.001, int epochs = 5000)
        : Optimizer(type), start(std::move(start)), lr(lr), epochs(epochs) {}

    VectorData optimize(ObjectiveFunction f, size_t /*dim*/) override {
        VectorData current = start;
        double sign = (type == OptimizationType::MINIMIZE) ? -1.0 : 1.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            VectorData grad = numerical_gradient(f, current);
            for (size_t i = 0; i < current.size(); ++i) {
                current[i] += sign * lr * grad[i];
            }
        }
        return current;
    }
};

// Генетический алгоритм (для непрерывных значений)
class GeneticAlgorithmOptimizer : public Optimizer {
    int pop_size;
    int epochs;

public:
    GeneticAlgorithmOptimizer(OptimizationType type,
                              int pop_size = 100, int epochs = 500)
        : Optimizer(type), pop_size(pop_size), epochs(epochs) {}

    VectorData optimize(ObjectiveFunction f, size_t dim) override {
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

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Оценка фитнеса
            std::vector<std::pair<double, VectorData>> fitness;
            for (const auto& ind : population) {
                fitness.push_back({evaluate(ind), ind});
            }

            // Сортировка по убыванию фитнеса
            std::sort(fitness.begin(), fitness.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });

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
        }

        // Возвращаем лучшего индивида
        double best_fitness = -1e9;
        VectorData best_ind;
        for (const auto& ind : population) {
            double fit = evaluate(ind);
            if (fit > best_fitness) {
                best_fitness = fit;
                best_ind = ind;
            }
        }
        return best_ind;
    }
};
