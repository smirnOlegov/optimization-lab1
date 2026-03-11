#pragma once

#include "types.h"
#include <cassert>
#include <cmath>

inline VectorData vec_sub(const VectorData& a, const VectorData& b) {
    assert(a.size() == b.size());
    VectorData r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}

inline VectorData vec_add(const VectorData& a, const VectorData& b) {
    assert(a.size() == b.size());
    VectorData r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

inline VectorData vec_scale(double s, const VectorData& v) {
    VectorData r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = s * v[i];
    return r;
}

inline double vec_dot(const VectorData& a, const VectorData& b) {
    assert(a.size() == b.size());
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline double vec_norm_sq(const VectorData& v) {
    return vec_dot(v, v);
}

inline VectorData vec_linear_combo(double a, const VectorData& u,
                                   double b, const VectorData& v) {
    assert(u.size() == v.size());
    VectorData r(u.size());
    for (size_t i = 0; i < u.size(); ++i) r[i] = a * u[i] + b * v[i];
    return r;
}
