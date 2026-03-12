#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <initializer_list>
#include <cmath>

// Рациональное число (Множество Q)
class Rational {
public:
    int64_t num, den;
    void reduce() {
        if (den == 0) den = 1; // Спасает от краша

        int64_t gcd = std::gcd(num, den);
        num /= gcd; den /= gcd;
        if (den < 0) { num = -num; den = -den; }

        // Безопасный предел 2 миллиарда (чтобы квадрат помещался в int64_t)
        const int64_t MAX_SAFE = 2000000000LL;
        while (std::abs(num) > MAX_SAFE || std::abs(den) > MAX_SAFE) {
            num /= 10;
            den /= 10;
            if (den == 0) den = 1;
        }
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

    operator double() const { return static_cast<double>(num) / den; }
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
        a1 = left->a1 - right->a2;
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
        : RealNode(Rational(0, 1), Rational(0, 1)), left(l), right(r) { update_bounds(); }

    void refine(Rational target_delta) override {
        while (delta > target_delta) {
            Rational old_delta = delta; // Для защиты от зависания

            Rational l_half = left->delta / Rational(2, 1);
            Rational r_half = right->delta / Rational(2, 1);
            if (l_half == Rational(0,1)) l_half = Rational(1,10);
            if (r_half == Rational(0,1)) r_half = Rational(1,10);

            if (left->delta > Rational(0,1)) left->refine(l_half);
            if (right->delta > Rational(0,1)) right->refine(r_half);

            update_bounds();

            // Если дельта не изменилась (достигнут лимит Rational), выходим
            if (delta == old_delta) break;
        }
    }
};

// ДЕЛЕНИЕ
class DivNode : public RealNode {
    std::shared_ptr<RealNode> left, right;

    void update_bounds() {
        if (right->a1 <= Rational(0, 1) && Rational(0, 1) <= right->a2) {
            // Если интервал знаменателя уже достаточно узкий, но всё ещё содержит 0
            // (значит число в реальности равно 0), мы искусственно ��двигаем его,
            // чтобы избежать деления на 0 и бесконечных циклов.
            if (right->delta < Rational(1, 100000)) { // Меньше 1e-5
                // Подменяем знаменатель на крошечное положительное число
                Rational tiny(1, 100000);
                Rational inv_a1 = Rational(1, 1) / tiny;
                Rational inv_a2 = Rational(1, 1) / tiny;

                std::initializer_list<Rational> gamma = {
                    left->a1 * inv_a1, left->a2 * inv_a1,
                    left->a1 * inv_a2, left->a2 * inv_a2
                };
                a1 = std::min(gamma);
                a2 = std::max(gamma);
                delta = a2 - a1;
                return;
            }

            // Если интервал ещё широкий, просто возвращаем безопасный "широкий" результат,
            // чтобы refine() мог продолжить сужать его дальше.
            a1 = Rational(-1000000000LL, 1);
            a2 = Rational(1000000000LL, 1);
            delta = a2 - a1;
            return;
        }

        Rational inv_a1 = Rational(1, 1) / right->a2;
        Rational inv_a2 = Rational(1, 1) / right->a1;

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
        : RealNode(Rational(0, 1), Rational(0, 1)), left(l), right(r) { update_bounds(); }

    void refine(Rational target_delta) override {
        while (delta > target_delta) {
            Rational old_delta = delta;

            Rational l_half = left->delta / Rational(2, 1);
            Rational r_half = right->delta / Rational(2, 1);
            if (l_half == Rational(0,1)) l_half = Rational(1,10);
            if (r_half == Rational(0,1)) r_half = Rational(1,10);

            if (left->delta > Rational(0,1)) left->refine(l_half);
            if (right->delta > Rational(0,1)) right->refine(r_half);

            update_bounds();
            if (delta == old_delta) break; // Защита от вечного цикла
        }
    }
};

class ExpNode : public RealNode {
private:
    std::shared_ptr<RealNode> arg;

public:
    ExpNode(std::shared_ptr<RealNode> a) : RealNode(Rational(0,1), Rational(0,1)), arg(a) {
        update_bounds();
    }

    void update_bounds() {
        double a1_d = static_cast<double>(arg->a1);
        double a2_d = static_cast<double>(arg->a2);

        double exp_a1 = std::exp(a1_d);
        double exp_a2 = std::exp(a2_d);

        long long MULT = 1000000LL;
        long long l_bound = static_cast<long long>(std::floor(exp_a1 * 0.999999 * MULT));
        long long r_bound = static_cast<long long>(std::ceil(exp_a2 * 1.000001 * MULT));

        this->a1 = Rational(l_bound, MULT);
        this->a2 = Rational(r_bound, MULT);
        this->delta = this->a2 - this->a1;
    }

    void refine(Rational target_delta) override {
        while (this->delta > target_delta) {
            Rational old_delta = delta;
            if (arg->delta > Rational(0,1)) {
                arg->refine(arg->delta / Rational(2, 1));
            }
            update_bounds();
            if (delta == old_delta) break;
        }
    }
};

class LogNode : public RealNode {
private:
    std::shared_ptr<RealNode> arg;

public:
    LogNode(std::shared_ptr<RealNode> a) : RealNode(Rational(0,1), Rational(0,1)), arg(a) {
        update_bounds();
    }

    void update_bounds() {
        double a1_d = static_cast<double>(arg->a1);
        double a2_d = static_cast<double>(arg->a2);

        if (a1_d <= 0.0) a1_d = 1e-9;
        if (a2_d <= 0.0) a2_d = 1e-8;

        double log_a1 = std::log(a1_d);
        double log_a2 = std::log(a2_d);

        double bound1 = (log_a1 > 0) ? log_a1 * 0.999999 : log_a1 * 1.000001;
        double bound2 = (log_a2 > 0) ? log_a2 * 1.000001 : log_a2 * 0.999999;

        long long MULT = 1000000LL;
        this->a1 = Rational(static_cast<long long>(std::floor(bound1 * MULT)), MULT);
        this->a2 = Rational(static_cast<long long>(std::ceil(bound2 * MULT)), MULT);
        this->delta = this->a2 - this->a1;
    }

    void refine(Rational target_delta) override {
        while (this->delta > target_delta) {
            Rational old_delta = delta;
            if (arg->delta > Rational(0,1)) {
                arg->refine(arg->delta / Rational(2, 1));
            }
            update_bounds();
            if (delta == old_delta) break;
        }
    }
};

// Обёртка
class ConstructiveReal {
private:
    std::shared_ptr<RealNode> root;
    ConstructiveReal(std::shared_ptr<RealNode> node) : root(node) {}

public:
    ConstructiveReal() : root(std::make_shared<ConstNode>(Rational(0, 1))) {}

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
    ConstructiveReal exp() const {
        return {std::make_shared<ExpNode>(this->root)};
    }
    ConstructiveReal log() const {
        return {std::make_shared<LogNode>(this->root)};
    }

    double to_double() const {
        return static_cast<double>(this->root->a1);
    }

    void collapse() {
        // Уточняем дерево до высокой точности (1e-8)
        Rational target(1, 100000000LL);
        this->root->refine(target);

        // Берем вычисленное значение
        double val = static_cast<double>(this->root->a1);

        // Превращаем в точную константу со знаменателем 10^8
        long long new_den = 100000000LL;
        long long new_num = std::round(val * new_den);

        // Заменяем дерево, очищая память
        this->root = std::make_shared<ConstNode>(Rational(new_num, new_den));
    }

    void print(const std::string& name) const {
        std::cout << name << " = [" << root->a1 << ", " << root->a2 << "], delta = " << root->delta << "\n";
    }

    bool operator<(const ConstructiveReal& other) const {
        Rational current_precision(1, 10);

        // Максимум 25 сужений интервала. Если за 25 итераций (точность ~3e-8)
        // интервалы так и не разошлись, значит числа практически равны.
        for (int i = 0; i < 25; ++i) {
            if (this->root->a2 < other.root->a1) return true;
            if (other.root->a2 < this->root->a1) return false;

            current_precision = current_precision / Rational(2, 1);
            this->root->refine(current_precision);
            other.root->refine(current_precision);
        }

        // Предотвращает зависание генетического алгоритма
        return false;
    }
};