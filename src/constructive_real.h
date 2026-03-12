#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <initializer_list>

// Рациональное число (Множество Q)
class Rational {
public:
    int64_t num, den;
    void reduce() {
        if (den == 0) throw std::invalid_argument("Division by zero!");
        int64_t gcd = std::gcd(num, den);
        num /= gcd; den /= gcd;
        if (den < 0) { num = -num; den = -den; }
    }
    Rational(int64_t n = 0, int64_t d = 1) : num(n), den(d) { reduce(); }

    Rational operator+(const Rational& o) const { return {num*o.den + o.num*den, den*o.den}; }
    Rational operator-(const Rational& o) const { return {num*o.den - o.num*den, den*o.den}; }
    Rational operator*(const Rational& o) const { return {num*o.num, den*o.den}; }
    Rational operator/(const Rational& o) const { return {num*o.den, den*o.num}; }

    bool operator<(const Rational& o) const { return num*o.den < o.num*den; }
    bool operator>(const Rational& o) const { return o < *this; }
    bool operator<=(const Rational& o) const { return !(o < *this); }
    bool operator==(const Rational& o) const { return num*o.den == o.num*den; }
    bool operator!=(const Rational& o) const { return !(*this == o); }

    friend std::ostream& operator<<(std::ostream& os, const Rational& r) {
        return (r.den == 1) ? (os << r.num) : (os << r.num << "/" << r.den);
    }
};

// Базовый Узел (RealNode)
class RealNode {
public:
    Rational a1, a2, delta; // a_down, a_up и разница между ними
    RealNode(Rational low, Rational high) : a1(low), a2(high), delta(high - low) {}
    virtual ~RealNode() = default;
    virtual void refine(Rational target_delta) = 0;
};

// Константа
class ConstNode : public RealNode {
public:
    ConstNode(Rational exact) : RealNode(exact, exact) {}
    void refine(Rational target_delta) override {} // Константа всегда точная
};

// Базовый генератор
class GeneratorNode : public RealNode {
    std::function<std::pair<Rational, Rational>(Rational)> gen_func;
public:
    GeneratorNode(Rational start_a1, Rational start_a2,
                  std::function<std::pair<Rational, Rational>(Rational)> func)
        : RealNode(start_a1, start_a2), gen_func(func) {}

    void refine(Rational target_delta) override {
        if (delta <= target_delta) return;

        // Без фишек C++17 (structured bindings), используем классический std::pair
        std::pair<Rational, Rational> new_bounds = gen_func(target_delta);
        a1 = new_bounds.first;
        a2 = new_bounds.second;
        delta = a2 - a1;
    }
};

// СЛОЖЕНИЕ
class AddNode : public RealNode {
    std::shared_ptr<RealNode> left, right;
public:
    AddNode(std::shared_ptr<RealNode> l, std::shared_ptr<RealNode> r)
        : RealNode(l->a1 + r->a1, l->a2 + r->a2), left(l), right(r) {}

    void refine(Rational target_delta) override {
        if (delta <= target_delta) return;
        Rational half_delta = target_delta / Rational(2, 1);
        left->refine(half_delta);
        right->refine(half_delta);
        a1 = left->a1 + right->a1;
        a2 = left->a2 + right->a2;
        delta = a2 - a1;
    }
};

// ВЫЧИТАНИЕ
class SubNode : public RealNode {
    std::shared_ptr<RealNode> left, right;
public:
    SubNode(std::shared_ptr<RealNode> l, std::shared_ptr<RealNode> r)
        : RealNode(l->a1 - r->a2, l->a2 - r->a1), left(l), right(r) {}

    void refine(Rational target_delta) override {
        if (delta <= target_delta) return;
        Rational half_delta = target_delta / Rational(2, 1);
        left->refine(half_delta);
        right->refine(half_delta);
        a1 = left->a1 - right->a2; // Крест-накрест из лекции
        a2 = left->a2 - right->a1;
        delta = a2 - a1;
    }
};

// УМНОЖЕНИЕ
class MulNode : public RealNode {
    std::shared_ptr<RealNode> left, right;

    void update_bounds() {
        std::initializer_list<Rational> gamma = {
            left->a1 * right->a1, left->a2 * right->a1,
            left->a1 * right->a2, left->a2 * right->a2
        };
        a1 = std::min(gamma);
        a2 = std::max(gamma);
        delta = a2 - a1;
    }

public:
    MulNode(std::shared_ptr<RealNode> l, std::shared_ptr<RealNode> r)
        : RealNode(Rational(0), Rational(0)), left(l), right(r) { update_bounds(); }

    void refine(Rational target_delta) override {
        // Умножение сложно предсказать, поэтому мы в цикле сужаем детей,
        // пока наше собственное delta не станет меньше target_delta
        while (delta > target_delta) {
            Rational l_half = left->delta / Rational(2, 1);
            Rational r_half = right->delta / Rational(2, 1);
            if (l_half == Rational(0,1)) l_half = Rational(1,10); // Заглушка, если левое было точным
            if (r_half == Rational(0,1)) r_half = Rational(1,10);

            if (left->delta > Rational(0,1)) left->refine(l_half);
            if (right->delta > Rational(0,1)) right->refine(r_half);

            update_bounds();
        }
    }
};

// ДЕЛЕНИЕ
class DivNode : public RealNode {
    std::shared_ptr<RealNode> left, right;

    void update_bounds() {
        // Проверка условия из лекции: если 0 внутри знаменателя, делить нельзя
        if (right->a1 <= Rational(0) && Rational(0) <= right->a2) {
            throw std::runtime_error("Division by an interval containing zero!");
        }

        // beta_inv = (1/b_up, 1/b_down)
        Rational inv_a1 = Rational(1) / right->a2;
        Rational inv_a2 = Rational(1) / right->a1;

        std::initializer_list<Rational> gamma = {
            left->a1 * inv_a1, left->a2 * inv_a1,
            left->a1 * inv_a2, left->a2 * inv_a2
        };
        a1 = std::min(gamma);
        a2 = std::max(gamma);
        delta = a2 - a1;
    }

public:
    DivNode(std::shared_ptr<RealNode> l, std::shared_ptr<RealNode> r)
        : RealNode(Rational(0), Rational(0)), left(l), right(r) { update_bounds(); }

    void refine(Rational target_delta) override {
        while (delta > target_delta) {
            Rational l_half = left->delta / Rational(2, 1);
            Rational r_half = right->delta / Rational(2, 1);
            if (l_half == Rational(0,1)) l_half = Rational(1,10);
            if (r_half == Rational(0,1)) r_half = Rational(1,10);

            if (left->delta > Rational(0,1)) left->refine(l_half);
            if (right->delta > Rational(0,1)) right->refine(r_half);

            update_bounds();
        }
    }
};

// Обёртка
class ConstructiveReal {
private:
    std::shared_ptr<RealNode> root;
    ConstructiveReal(std::shared_ptr<RealNode> node) : root(node) {}

public:
    ConstructiveReal(Rational exact) : root(std::make_shared<ConstNode>(exact)) {}

    ConstructiveReal(Rational start_a1, Rational start_a2,
                     std::function<std::pair<Rational, Rational>(Rational)> func)
        : root(std::make_shared<GeneratorNode>(start_a1, start_a2, func)) {}

    ConstructiveReal operator+(const ConstructiveReal& other) const {
        return {std::make_shared<AddNode>(this->root, other.root)};
    }
    ConstructiveReal operator-(const ConstructiveReal& other) const {
        return {std::make_shared<SubNode>(this->root, other.root)};
    }
    ConstructiveReal operator*(const ConstructiveReal& other) const {
        return {std::make_shared<MulNode>(this->root, other.root)};
    }
    ConstructiveReal operator/(const ConstructiveReal& other) const {
        return {std::make_shared<DivNode>(this->root, other.root)};
    }


    void print(const std::string& name) const {
        std::cout << name << " = [" << root->a1 << ", " << root->a2 << "], delta = " << root->delta << "\n";
    }

    // Оператор сравнения (с автоматическим уточнением!)
    bool operator<(const ConstructiveReal& other) const {
        Rational current_precision = std::max(this->root->delta, other.root->delta);
        if (current_precision == Rational(0,1)) current_precision = Rational(1,1);

        while (true) {
            if (this->root->a2 < other.root->a1) return true;
            if (other.root->a2 < this->root->a1) return false;

            current_precision = current_precision / Rational(2, 1);
            this->root->refine(current_precision);
            other.root->refine(current_precision);
        }
    }
};
