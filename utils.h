#pragma once

#include "types.h"
#include <iostream>
#include <iomanip>
#include <string>

void print_result(const std::string& title, const VectorData& res, double val) {
    std::cout << "--- " << title << " ---\n";
    std::cout << "Point: [";
    for (size_t i = 0; i < res.size(); ++i) {
        std::cout << std::fixed << std::setprecision(5) << res[i] << (i == res.size() - 1 ? "" : ", ");
    }
    std::cout << "]\nValue: " << val << "\n\n";
}
