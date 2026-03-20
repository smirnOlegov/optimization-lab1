#pragma once

#include <functional>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <algorithm>
#include <initializer_list>
#include <cmath>
#include <boost/multiprecision/cpp_int.hpp>

// Рациональное число (Множество Q)
using Rational = boost::multiprecision::cpp_rational;

// Базовый Узел (RealNode)
class RealNode {
public:
    Rational a1, a2, delta; // a_down, a_up и разница между ними
    RealNode(Rational low, Rational high) : a1(low), a2(high), delta(high - low) {}
    virtual ~RealNode() = default;
    virtual void refine(Rational target_delta) = 0;
    virtual bool is_exact() const { return false; }
};

// Константа
class ConstNode : public RealNode {
public:
    ConstNode(Rational exact) : RealNode(exact, exact) {}
    void refine(Rational target_delta) override {} // Константа всегда точная
    bool is_exact() const override { return true; }
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

    bool is_exact() const override { return false; }
};

// Сложение
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

// Замороженный интервал (результат коллапса)
class IntervalNode : public RealNode {
public:
    IntervalNode(Rational low, Rational high) : RealNode(low, high) {}

    void refine(Rational target_delta) override {
        if (target_delta < delta) { }
    }

    bool is_exact() const override { return false; }
};

// Вычитание
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

// Умножение
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

            if (delta == old_delta) break;
        }
    }
};

// Деление
class DivNode : public RealNode {
    std::shared_ptr<RealNode> left, right;

    void update_bounds() {
        if (right->a1 <= Rational(0, 1) && Rational(0, 1) <= right->a2) {
            if (right->delta < Rational(1, 100000)) { // Меньше 1e-5
                // Подменяем знаменатель на крошечное положительное число если число ~ 0
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
        if (this->root->is_exact() && other.root->is_exact()) {
            return ConstructiveReal(this->root->a1 + other.root->a1);
        }
        return {std::make_shared<AddNode>(this->root, other.root)};
    }

    ConstructiveReal operator-(const ConstructiveReal& other) const {
        if (this->root->is_exact() && other.root->is_exact()) {
            return ConstructiveReal(this->root->a1 - other.root->a1);
        }
        return {std::make_shared<SubNode>(this->root, other.root)};
    }

    ConstructiveReal operator*(const ConstructiveReal& other) const {
        if (this->root->is_exact() && other.root->is_exact()) {
            return ConstructiveReal(this->root->a1 * other.root->a1);
        }
        return {std::make_shared<MulNode>(this->root, other.root)};
    }

    ConstructiveReal operator/(const ConstructiveReal& other) const {
        if (this->root->is_exact() && other.root->is_exact()) {
            // Защита от деления на 0
            if (other.root->a1 == 0) throw std::runtime_error("Division by zero");
            return ConstructiveReal(this->root->a1 / other.root->a1);
        }
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

    void collapse(Rational target_precision = Rational(1, 1000000000LL)) {
        this->root->refine(target_precision);

        Rational low = this->root->a1;
        Rational high = this->root->a2;

        // Округление
        Rational center = (low + high) / Rational(2, 1);
        double center_d = center.convert_to<double>(); // преобразуем в double (отбрасываем много цифр)

        // Превращаем обратно в легкую дробь с точностью 10^-9
        long long MULT = 1000000000LL;
        long long num = std::round(center_d * MULT);
        Rational simplified_center(num, MULT);

        // Расширяем интервал, чтобы покрыть ошибку округления (10^-9)
        Rational error_margin(1, MULT);
        Rational new_low = simplified_center - error_margin;
        Rational new_high = simplified_center + error_margin;

        this->root = std::make_shared<IntervalNode>(new_low, new_high);
    }

    void print(const std::string& name) const {
        double lower_bound = this->root->a1.convert_to<double>();
        double upper_bound = this->root->a2.convert_to<double>();
        double current_delta = this->root->delta.convert_to<double>();

        std::cout << name << " = ["
                  << std::fixed << std::setprecision(8) << lower_bound << ", "
                  << std::fixed << std::setprecision(8) << upper_bound << "], "
                  << "delta = " << std::scientific << current_delta << "\n";

        std::cout.unsetf(std::ios_base::floatfield);
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

        return false;
    }
};