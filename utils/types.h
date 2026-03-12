#pragma once

#include <vector>
#include <functional>
#include "../src/constructive_real.h"

using VectorData = std::vector<ConstructiveReal>;
using ObjectiveFunction = std::function<ConstructiveReal(const VectorData&)>;

enum class OptimizationType {
    MINIMIZE,
    MAXIMIZE
};