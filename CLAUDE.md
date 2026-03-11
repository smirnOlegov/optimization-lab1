# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
g++ -std=c++17 -O2 -o main main.cpp && ./main
```

Single-file build, no build system. All code is in headers (`.h`) included by `main.cpp`. No tests or linter configured.

## Architecture

Header-only C++17 optimization framework with two independent components:

**Core types** (`types.h`): `VectorData` (alias for `std::vector<double>`), `ObjectiveFunction` (alias for `std::function<double(const VectorData&)>`), `OptimizationType` enum (MINIMIZE/MAXIMIZE).

**Optimizer hierarchy** (`optimizer.h`): Abstract base `Optimizer` with virtual `optimize(ObjectiveFunction f, size_t dim) -> VectorData`. Concrete subclasses:
- `GradientDescentOptimizer` — fixed learning rate, numerical gradients
- `GeneticAlgorithmOptimizer` — population-based, continuous-domain

**Accelerated GRAAL** (`accel_graal_optimizer.h`): Adaptive gradient method with Nesterov acceleration (arXiv:2507.09823). Uses `vector_ops.h` for vector arithmetic. Handles non-convex objectives by falling back to growing stepsize when curvature estimates go negative, and returns the best iterate seen.

**ConstructiveReal** (`constructive_real.h`): Lazy-evaluated computable reals — independent from the optimizer hierarchy.

**Utilities**: `numerical_gradient.h` (central differences), `objective_functions.h` (Rosenbrock 2D/ND), `vector_ops.h` (inline vector arithmetic), `utils.h` (result printing).

## Conventions

- Comments and variable descriptions are in Russian.
- New optimizers inherit from `Optimizer` and implement `optimize()`.
- Objective functions take `const VectorData&` and return `double`.
- Numerical gradients via central differences (`numerical_gradient.h`) — no symbolic/automatic differentiation.
