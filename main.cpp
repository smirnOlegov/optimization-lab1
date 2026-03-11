#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <random>
#include <algorithm>
#include <iomanip>

// ==========================================
// 1. Конструктивное действительное число
// ==========================================
// В вычислимом анализе конструктивное число - это функция, 
// которая для любой заданной точности epsilon возвращает рациональное (здесь double) приближение.
class ConstructiveReal {
private:
    std::function<double(double)> generator;

public:
    // Конструктор от генератора (вычислимой функции)
    ConstructiveReal(std::function<double(double)> gen) : generator(gen) {}

    // Конструктор от константы
    ConstructiveReal(double value) {
        generator = [value](double eps) { return value; };
    }

    // Вычисление с заданной точностью
    double evaluate(double epsilon = 1e-7) const {
        return generator(epsilon);
    }

    // Пример арифметических операций над конструктивными числами (ленивое вычисление)
    ConstructiveReal operator+(const ConstructiveReal& other) const {
        auto genA = this->generator;
        auto genB = other.generator;
        return ConstructiveReal([genA, genB](double eps) {
            return genA(eps / 2.0) + genB(eps / 2.0);
        });
    }

    ConstructiveReal operator*(const ConstructiveReal& other) const {
        auto genA = this->generator;
        auto genB = other.generator;
        return ConstructiveReal([genA, genB](double eps) {
            // Упрощенная оценка для умножения
            return genA(eps) * genB(eps);
        });
    }
};

// ==========================================
// 2. Определение Целевой Функции (Чёрный ящик)
// ==========================================
using VectorData = std::vector<double>;
using ObjectiveFunction = std::function<double(const VectorData&)>;

// ==========================================
// 3. Численное дифференцирование
// ==========================================
VectorData numerical_gradient(const ObjectiveFunction& f, const VectorData& x, double h = 1e-5) {
    VectorData grad(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        VectorData x_plus = x;
        VectorData x_minus = x;
        
        x_plus[i] += h;
        x_minus[i] -= h;
        
        // Метод центральных разностей
        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * h);
    }
    return grad;
}

// ==========================================
// 4. Функции Розенброка
// ==========================================
double Rosenbrock2D(const VectorData& x) {
    if (x.size() < 2) return 0.0;
    // f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
    return std::pow(1.0 - x[0], 2) + 100.0 * std::pow(x[1] - std::pow(x[0], 2), 2);
}

double RosenbrockND(const VectorData& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; ++i) {
        sum += 100.0 * std::pow(x[i+1] - std::pow(x[i], 2), 2) + std::pow(1.0 - x[i], 2);
    }
    return sum;
}

// ==========================================
// 5. Алгоритмы Оптимизации
// ==========================================

// Общий перечислитель для задачи
enum class OptimizationType { MINIMIZE, MAXIMIZE };

// 5.1 Градиентный спуск (использует численное дифференцирование)
VectorData GradientDescent(ObjectiveFunction f, VectorData start, OptimizationType type, double lr = 0.001, int epochs = 5000) {
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

// 5.2 Генетический алгоритм (для непрерывных значений)
VectorData GeneticAlgorithm(ObjectiveFunction f, size_t dim, OptimizationType type, int pop_size = 100, int epochs = 500) {
    std::mt19937 gen(42); // Фиксированный сид для детерминированности/воспроизводимости
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
        return (type == OptimizationType::MINIMIZE) ? -val : val; // ГА всегда максимизирует фитнес
    };

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Оценка фитнеса
        std::vector<std::pair<double, VectorData>> fitness;
        for (const auto& ind : population) {
            fitness.push_back({evaluate(ind), ind});
        }

        // Сортировка по убыванию фитнеса (селекция лучших)
        std::sort(fitness.begin(), fitness.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        std::vector<VectorData> next_gen;
        // Элитизм: сохраняем топ 10%
        int elite_count = pop_size / 10;
        for (int i = 0; i < elite_count; ++i) {
            next_gen.push_back(fitness[i].second);
        }

        // Скрещивание (Арифметический кроссовер) и мутация
        std::uniform_int_distribution<> parent_dis(0, pop_size / 2); // выбираем из лучшей половины
        while (next_gen.size() < pop_size) {
            VectorData p1 = fitness[parent_dis(gen)].second;
            VectorData p2 = fitness[parent_dis(gen)].second;
            
            VectorData child(dim);
            for (size_t i = 0; i < dim; ++i) {
                // Скрещивание
                child[i] = (p1[i] + p2[i]) / 2.0;
                
                // Мутация (с вероятностью 20%)
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

// ==========================================
// Вспомогательная функция вывода
// ==========================================
void print_result(const std::string& title, const VectorData& res, double val) {
    std::cout << "--- " << title << " ---\n";
    std::cout << "Point: [";
    for (size_t i = 0; i < res.size(); ++i) {
        std::cout << std::fixed << std::setprecision(5) << res[i] << (i == res.size() - 1 ? "" : ", ");
    }
    std::cout << "]\nValue: " << val << "\n\n";
}

int main() {
    // 1. Тест конструктивного числа
    ConstructiveReal cr1([](double eps) { return 3.1415926535; }); // Приближение PI
    ConstructiveReal cr2(2.0);
    ConstructiveReal cr3 = cr1 * cr2;
    std::cout << "Constructive Real Test (PI * 2) = " << cr3.evaluate(1e-5) << "\n\n";

    // 2. Оптимизация 2D Розенброка (Минимум в [1, 1], значение 0)
    VectorData start_2d = {0.0, 0.0};
    
    std::cout << "==== ROSENBROCK 2D ====\n";
    VectorData gd_min_2d = GradientDescent(Rosenbrock2D, start_2d, OptimizationType::MINIMIZE, 0.002, 10000);
    print_result("Gradient Descent (MIN)", gd_min_2d, Rosenbrock2D(gd_min_2d));

    VectorData ga_min_2d = GeneticAlgorithm(Rosenbrock2D, 2, OptimizationType::MINIMIZE, 200, 500);
    print_result("Genetic Algorithm (MIN)", ga_min_2d, Rosenbrock2D(ga_min_2d));

    // Максимизация (искусственная задача для Розенброка - "убежать" от минимума)
    // ГА отлично справится с поиском максимума на заданном отрезке мутаций
    VectorData ga_max_2d = GeneticAlgorithm(Rosenbrock2D, 2, OptimizationType::MAXIMIZE, 200, 500);
    print_result("Genetic Algorithm (MAX - Divergence Demo)", ga_max_2d, Rosenbrock2D(ga_max_2d));



    // 3. Оптимизация многомерной функции Розенброка (N=5)
    std::cout << "==== ROSENBROCK ND (N=5) ====\n";
    VectorData start_nd = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    VectorData gd_min_nd = GradientDescent(RosenbrockND, start_nd, OptimizationType::MINIMIZE, 0.001, 15000);
    print_result("Gradient Descent ND (MIN)", gd_min_nd, RosenbrockND(gd_min_nd));

    VectorData ga_min_nd = GeneticAlgorithm(RosenbrockND, 5, OptimizationType::MINIMIZE, 500, 1000);
    print_result("Genetic Algorithm ND (MIN)", ga_min_nd, RosenbrockND(ga_min_nd));

    return 0;
}
