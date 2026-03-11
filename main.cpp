#include <iostream>
#include <chrono>

#include "constructive_real.h"
#include "objective_functions.h"
#include "optimizer.h"
#include "accel_graal_optimizer.h"
#include "utils.h"

int main() {
    // 1. Тест конструктивного числа
    ConstructiveReal cr1([](double eps) { return 3.1415926535; }); // Приближение PI
    ConstructiveReal cr2(2.0);
    ConstructiveReal cr3 = cr1 * cr2;
    std::cout << "Constructive Real Test (PI * 2) = " << cr3.evaluate(1e-5) << "\n\n";

    // 2. Оптимизация 2D Розенброка (Минимум в [1, 1], значение 0)
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

    // 3. Оптимизация многомерной функции Розенброка (N=5)
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
