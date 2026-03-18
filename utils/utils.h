#pragma once

#include "types.h"
#include <iostream>
#include <string>

inline void print_result(const std::string& title, const VectorData& res, ConstructiveReal val,
                  size_t iterations, ConstructiveReal target_value, bool converged) {
    std::cout << "--- " << title << " ---\n";
    std::cout << "Point:\n";

    for (size_t i = 0; i < res.size(); ++i) {
        res[i].print("  x[" + std::to_string(i) + "]");
    }

    val.print("Value");

    std::cout << "Iterations: " << iterations << "\n";

    ConstructiveReal diff = (val < target_value) ? (target_value - val) : (val - target_value);
    diff.print("Accuracy");

    std::cout << "Status: " << (converged ? "target reached" : "max iterations reached")
              << "\n\n";
}