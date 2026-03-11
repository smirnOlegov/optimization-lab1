#include <iostream>

#include "constructive_real.h"
#include "objective_functions.h"
#include "optimizer.h"
#include "utils.h"

int main() {
    // 1. Тест конструктивного числа
    ConstructiveReal cr1([](double eps) { return 3.1415926535; }); // Приближение PI
    ConstructiveReal cr2(2.0);
    ConstructiveReal cr3 = cr1 * cr2;
    std::cout << "Constructive Real Test (PI * 2) = " << cr3.evaluate(1e-5) << "\n\n";

    // 2. Оптимизация 2D Розенброка (Минимум в [1, 1], значение 0)
    VectorData start_2d = {0.0, 0.0};

    std::cout << "==== ROSENBROCK 2D ====\n";

    GradientDescentOptimizer gd_2d(start_2d, OptimizationType::MINIMIZE, 0.002, 10000);
    VectorData gd_min_2d = gd_2d.optimize(Rosenbrock2D, 2);
    print_result("Gradient Descent (MIN)", gd_min_2d, Rosenbrock2D(gd_min_2d));

    GeneticAlgorithmOptimizer ga_min(OptimizationType::MINIMIZE, 200, 500);
    VectorData ga_min_2d = ga_min.optimize(Rosenbrock2D, 2);
    print_result("Genetic Algorithm (MIN)", ga_min_2d, Rosenbrock2D(ga_min_2d));

    GeneticAlgorithmOptimizer ga_max(OptimizationType::MAXIMIZE, 200, 500);
    VectorData ga_max_2d = ga_max.optimize(Rosenbrock2D, 2);
    print_result("Genetic Algorithm (MAX - Divergence Demo)", ga_max_2d, Rosenbrock2D(ga_max_2d));

    // 3. Оптимизация многомерной функции Розенброка (N=5)
    std::cout << "==== ROSENBROCK ND (N=5) ====\n";
    VectorData start_nd = {0.0, 0.0, 0.0, 0.0, 0.0};

    GradientDescentOptimizer gd_nd(start_nd, OptimizationType::MINIMIZE, 0.001, 15000);
    VectorData gd_min_nd = gd_nd.optimize(RosenbrockND, 5);
    print_result("Gradient Descent ND (MIN)", gd_min_nd, RosenbrockND(gd_min_nd));

    GeneticAlgorithmOptimizer ga_nd(OptimizationType::MINIMIZE, 500, 1000);
    VectorData ga_min_nd = ga_nd.optimize(RosenbrockND, 5);
    print_result("Genetic Algorithm ND (MIN)", ga_min_nd, RosenbrockND(ga_min_nd));

    return 0;
}
