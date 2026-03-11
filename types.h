#pragma once

#include <vector>
#include <functional>

using VectorData = std::vector<double>;
using ObjectiveFunction = std::function<double(const VectorData&)>;

enum class OptimizationType { MINIMIZE, MAXIMIZE };
