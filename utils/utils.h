#pragma once

#include "types.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

void print_result(const std::string& title, const VectorData& res, double val,
                  size_t iterations, double target_value, bool converged) {
    std::cout << "--- " << title << " ---\n";
    std::cout << "Point: [";
    for (size_t i = 0; i < res.size(); ++i) {
        std::cout << std::fixed << std::setprecision(5) << res[i] << (i == res.size() - 1 ? "" : ", ");
    }
    std::cout << "]\nValue: " << val
              << "\nIterations: " << iterations
              << "\nAccuracy: " << std::abs(val - target_value)
              << "\nStatus: " << (converged ? "target reached" : "max iterations reached")
              << "\n\n";
}
