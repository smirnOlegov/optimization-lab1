#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>

#include "src/constructive_real.h"
#include "src/objective_functions.h"
#include "optimizers/gradient_descent.h"
#include "optimizers/genetic_algo.h"
#include "optimizers/accel_graal_optimizer.h"
#include "utils/utils.h"

void save_history_to_csv(const std::string& filename, const std::vector<std::vector<double>>& history) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Ошибка: Не удалось открыть файл " << filename << " для записи!\n";
        return;
    }

    if (!history.empty()) {
        for (size_t i = 0; i < history[0].size(); ++i) {
            file << "x" << i;
            if (i < history[0].size() - 1) file << ",";
        }
        file << "\n";
    }

    for (const auto& point : history) {
        for (size_t i = 0; i < point.size(); ++i) {
            file << point[i]; // Здесь убрали to_double(), так как point[i] - это уже double!
            if (i < point.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

int main() {
    // Тест конструктивного числа
    std::cout << "==== CONSTRUCTIVE REAL TEST ====\n\n";

    // Создаем числа на основе точных дробей
    ConstructiveReal cr1(Rational(22, 7)); // Приближение PI (22/7)
    ConstructiveReal cr2(Rational(2, 1));  // Точное число 2/1

    // Тестируем все новые интервальные операции
    ConstructiveReal cr_add = cr1 + cr2;
    ConstructiveReal cr_sub = cr1 - cr2;
    ConstructiveReal cr_mul = cr1 * cr2;
    ConstructiveReal cr_div = cr1 / cr2;

    cr_add.print("PI_approx + 2");
    cr_sub.print("PI_approx - 2");
    cr_mul.print("PI_approx * 2");
    cr_div.print("PI_approx / 2");

    // 1. ПРОДВИНУТЫЙ Тест конструктивного числа
    std::cout << "\n\n==== ADVANCED CONSTRUCTIVE REAL TEST ====\n\n";

    // Генератор корня из 2 (Бинарный поиск на дробях) ---
    auto sqrt2_gen = [](Rational target_delta) {
        Rational L(1, 1); // Нижняя граница (1)
        Rational R(2, 1); // Верхняя граница (2)
        Rational two(2, 1);

        // Пока ширина интервала больше требуемой, сужаем его пополам
        while (target_delta < (R - L)) {
            Rational M = (L + R) / Rational(2, 1);
            if ((M * M) < two) L = M; // Если M^2 < 2, то корень правее
            else R = M;               // Иначе корень левее
        }
        return std::make_pair(L, R);
    };

    // Создаем конструктивное число sqrt(2). Изначально мы лишь знаем, что оно в [1, 2].
    ConstructiveReal Sqrt2(Rational(1, 1), Rational(2, 1), sqrt2_gen);

    std::cout << "[Test 1] Sqrt(2) boundaries check:\n";
    ConstructiveReal low_bound_sqrt(Rational(1414, 1000));  // 1.414
    ConstructiveReal high_bound_sqrt(Rational(1415, 1000)); // 1.415

    // В этот момент сработает refine() и бинарный поиск вычислит нужную точность!
    std::cout << "Is 1.414 < sqrt(2)? " << (low_bound_sqrt < Sqrt2 ? "Yes" : "No") << "\n";
    std::cout << "Is sqrt(2) < 1.415? " << (Sqrt2 < high_bound_sqrt ? "Yes" : "No") << "\n\n";

    // --- Б. Вычисления с иррациональными числами ---
    std::cout << "[Test 2] Sqrt(2) * Sqrt(2) approx 2:\n";
    ConstructiveReal Sqrt2_Squared = Sqrt2 * Sqrt2; // Строим дерево: MulNode(Sqrt2, Sqrt2)

    ConstructiveReal low_two(Rational(199, 100)); // 1.99
    ConstructiveReal high_two(Rational(201, 100)); // 2.01

    // Чтобы сравнить квадрат корня с 1.99, узлу умножения придется пнуть
    // генераторы корня, чтобы они выдали больше знаков после запятой.
    std::cout << "Is 1.99 < (sqrt(2) * sqrt(2))? " << (low_two < Sqrt2_Squared ? "Yes" : "No") << "\n";
    std::cout << "Is (sqrt(2) * sqrt(2)) < 2.01? " << (Sqrt2_Squared < high_two ? "Yes" : "No") << "\n\n";

    // --- В. Генератор числа Эйлера (e) через ряд Тейлора ---
    auto e_gen = [](Rational target_delta) {
        Rational L(2, 1); // Первые члены: 1/0! + 1/1! = 2
        int64_t k = 1;
        int64_t fact = 1;

        // Оценка остатка ряда. Верхняя граница R = L + 1/(k * k!)
        Rational R = L + Rational(1, fact * k);

        while (target_delta < (R - L)) {
            k++;
            fact *= k;
            L = L + Rational(1, fact); // Добавляем следующий член ряда
            R = L + Rational(1, fact * (k - 1)); // Обновляем верхнюю границу
        }
        return std::make_pair(L, R);
    };

    // Изначально знаем, что e где-то между 2 и 3
    ConstructiveReal E_val(Rational(2, 1), Rational(3, 1), e_gen);

    std::cout << "[Test 3] Euler's number (e) boundaries check:\n";
    ConstructiveReal low_bound_e(Rational(271, 100)); // 2.71
    ConstructiveReal high_bound_e(Rational(272, 100)); // 2.72

    std::cout << "Is 2.71 < e? " << (low_bound_e < E_val ? "Yes" : "No") << "\n";
    std::cout << "Is e < 2.72? " << (E_val < high_bound_e ? "Yes" : "No") << "\n\n";

    ConstructiveReal zero_val(Rational(0, 1));

    // Оптимизация 2D Розенброка (Минимум в [1, 1], значение 0)
    VectorData start_2d = {zero_val, zero_val};
    ConstructiveReal rosenbrock_target = zero_val;
    ConstructiveReal tolerance_2d(Rational(1, 1000));
    ConstructiveReal tolerance_nd(Rational(1, 100));

    std::cout << "==== ROSENBROCK 2D ====\n\n";
    std::cout << "Target value: 0\n\n";

    struct TimedRun {
        std::string name;
        OptimizationResult result;
        double time_ms;
    };

    auto run_benchmark = [](const std::string& name, auto&& runner) {
        auto t0 = std::chrono::high_resolution_clock::now();
        OptimizationResult result = runner();
        auto t1 = std::chrono::high_resolution_clock::now();
        return TimedRun{
            name,
            result,
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        };
    };

    auto print_benchmark = [](const TimedRun& run, ConstructiveReal target_value) {
        print_result(run.name, run.result.point, run.result.value,
                     run.result.iterations, target_value, run.result.converged);
        std::cout << "Time: " << run.time_ms << " ms\n\n";
    };

    auto print_iteration_comparison = [](const std::string& title,
                                         const std::vector<TimedRun>& runs) {
        std::vector<const TimedRun*> ranking;
        for (const auto& run : runs) {
            if (run.result.converged) {
                ranking.push_back(&run);
            }
        }

        std::sort(ranking.begin(), ranking.end(), [](const TimedRun* lhs, const TimedRun* rhs) {
            return lhs->result.iterations < rhs->result.iterations;
        });

        std::cout << "Iteration comparison for " << title << ":\n";
        if (ranking.empty()) {
            std::cout << "No method reached the target accuracy.\n";
        } else {
            for (size_t i = 0; i < ranking.size(); ++i) {
                std::cout << (i + 1) << ". " << ranking[i]->name
                          << " - " << ranking[i]->result.iterations << " iterations\n";
            }
        }

        for (const auto& run : runs) {
            if (!run.result.converged) {
                std::cout << run.name << " did not reach the target accuracy within "
                          << run.result.iterations << " iterations\n";
            }
        }
        std::cout << "\n";
    };

    ConstructiveReal lr_001(Rational(1, 1000)); // 0.001
    ConstructiveReal lr_0005(Rational(5, 10000)); // 0.0005

    TimedRun gd_2d = run_benchmark("Gradient Descent", [&] {
        GradientDescentOptimizer gd(start_2d, OptimizationType::MINIMIZE, lr_001, 200000);
        OptimizationResult res = gd.optimize(Rosenbrock2D, 2, rosenbrock_target, tolerance_2d);

        // сохраняем историю в файл
        save_history_to_csv("visualisation/history/gd_history.csv", gd.history);

        return res; // Возвращаем результат
    });

    print_benchmark(gd_2d, rosenbrock_target);

    TimedRun ga_2d = run_benchmark("Genetic Algorithm", [&] {
        GeneticAlgorithmOptimizer ga(OptimizationType::MINIMIZE, 200, 5000);
        OptimizationResult res = ga.optimize(Rosenbrock2D, 2, rosenbrock_target, tolerance_2d);

        save_history_to_csv("visualisation/history/ga_history.csv", ga.history);

        return res;
    });
    print_benchmark(ga_2d, rosenbrock_target);

    ConstructiveReal eta_graal(Rational(5, 1000));   // 0.005
    ConstructiveReal theta_graal(Rational(2, 1));    // 2.0
    ConstructiveReal gamma_graal(Rational(1, 100));  // 0.01

    TimedRun graal_2d = run_benchmark("Accel GRAAL", [&] {
        AcceleratedGRAALOptimizer graal(start_2d, OptimizationType::MINIMIZE, 200000,
                                        eta_graal, theta_graal, gamma_graal);
        return graal.optimize(Rosenbrock2D, 2, rosenbrock_target, tolerance_2d);
    });

    print_benchmark(graal_2d, rosenbrock_target);

    print_iteration_comparison("Rosenbrock 2D", {gd_2d, ga_2d, graal_2d});

    // Оптимизация многомерной функции Розенброк�� (N=5)
    std::cout << "==== ROSENBROCK ND (N=5) ====\n\n";
    VectorData start_nd = {zero_val, zero_val, zero_val, zero_val, zero_val};
    std::cout << "Target value: 0\n\n";

    TimedRun gd_nd = run_benchmark("Gradient Descent", [&] {
        GradientDescentOptimizer gd(start_nd, OptimizationType::MINIMIZE, lr_0005, 300000);
        return gd.optimize(RosenbrockND, 5, rosenbrock_target, tolerance_nd);
    });
    print_benchmark(gd_nd, rosenbrock_target);

    TimedRun ga_nd = run_benchmark("Genetic Algorithm", [&] {
        // 1000 особей, 10000 поколений
        GeneticAlgorithmOptimizer ga(OptimizationType::MINIMIZE, 1000, 10000);
        return ga.optimize(RosenbrockND, 5, rosenbrock_target, tolerance_nd);
    });
    print_benchmark(ga_nd, rosenbrock_target);

    print_iteration_comparison("Rosenbrock ND (N=5)", {gd_nd, ga_nd});

    return 0;
}