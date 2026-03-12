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

inline VectorData vec_scale(ConstructiveReal s, const VectorData& v) {
    VectorData r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = s * v[i];
    return r;
}

inline ConstructiveReal vec_dot(const VectorData& a, const VectorData& b) {
    assert(a.size() == b.size());
    ConstructiveReal s = ConstructiveReal(Rational(0, 1));
    for (size_t i = 0; i < a.size(); ++i) s = s + a[i] * b[i];
    return s;
}

inline ConstructiveReal vec_norm_sq(const VectorData& v) {
    return vec_dot(v, v);
}

inline VectorData vec_linear_combo(ConstructiveReal a, const VectorData& u,
                                   ConstructiveReal b, const VectorData& v) {
    assert(u.size() == v.size());
    VectorData r(u.size());
    for (size_t i = 0; i < u.size(); ++i) r[i] = a * u[i] + b * v[i];
    return r;
}
