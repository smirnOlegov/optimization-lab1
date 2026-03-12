#include <iostream>
#include <chrono>

#include "constructive_real.h"
#include "objective_functions.h"
#include "optimizer.h"
#include "accel_graal_optimizer.h"
#include "utils.h"

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


    // Оптимизация 2D Розенброка (Минимум в [1, 1], значение 0)
    VectorData start_2d = {0.0, 0.0};

    std::cout << "==== ROSENBROCK 2D ====\n\n";

    double lrs[] = {0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01};
    for (double lr : lrs) {
        auto t0 = std::chrono::high_resolution_clock::now();
        GradientDescentOptimizer gd(start_2d, OptimizationType::MINIMIZE, lr, 10000);
        VectorData res = gd.optimize(Rosenbrock2D, 2);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::string label = "GD lr=" + std::to_string(lr);
        print_result(label, res, Rosenbrock2D(res));
        std::cout << "Time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n\n";
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    GeneticAlgorithmOptimizer ga_2d(OptimizationType::MINIMIZE, 200, 500);
    VectorData ga_min_2d = ga_2d.optimize(Rosenbrock2D, 2);
    auto t1 = std::chrono::high_resolution_clock::now();
    print_result("Genetic Algorithm", ga_min_2d, Rosenbrock2D(ga_min_2d));
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n\n";

    t0 = std::chrono::high_resolution_clock::now();
    AcceleratedGRAALOptimizer graal_2d(start_2d, OptimizationType::MINIMIZE, 50000, 0.001);
    VectorData graal_min_2d = graal_2d.optimize(Rosenbrock2D, 2);
    t1 = std::chrono::high_resolution_clock::now();
    print_result("Accel GRAAL", graal_min_2d, Rosenbrock2D(graal_min_2d));
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n\n";

    // Оптимизация многомерной функции Розенброка (N=5)
    std::cout << "==== ROSENBROCK ND (N=5) ====\n\n";
    VectorData start_nd = {0.0, 0.0, 0.0, 0.0, 0.0};

    for (double lr : lrs) {
        t0 = std::chrono::high_resolution_clock::now();
        GradientDescentOptimizer gd(start_nd, OptimizationType::MINIMIZE, lr, 15000);
        VectorData res = gd.optimize(RosenbrockND, 5);
        t1 = std::chrono::high_resolution_clock::now();
        std::string label = "GD lr=" + std::to_string(lr);
        print_result(label, res, RosenbrockND(res));
        std::cout << "Time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n\n";
    }

    t0 = std::chrono::high_resolution_clock::now();
    GeneticAlgorithmOptimizer ga_nd(OptimizationType::MINIMIZE, 500, 1000);
    VectorData ga_min_nd = ga_nd.optimize(RosenbrockND, 5);
    t1 = std::chrono::high_resolution_clock::now();
    print_result("Genetic Algorithm", ga_min_nd, RosenbrockND(ga_min_nd));
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n\n";

    t0 = std::chrono::high_resolution_clock::now();
    AcceleratedGRAALOptimizer graal_nd(start_nd, OptimizationType::MINIMIZE, 50000, 0.001);
    VectorData graal_min_nd = graal_nd.optimize(RosenbrockND, 5);
    t1 = std::chrono::high_resolution_clock::now();
    print_result("Accel GRAAL", graal_min_nd, RosenbrockND(graal_min_nd));
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n\n";

    return 0;
}