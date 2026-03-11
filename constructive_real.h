#pragma once

#include <functional>

// Конструктивное действительное число
// В вычислимом анализе конструктивное число - это функция,
// которая для любой заданной точности epsilon возвращает рациональное (здесь double) приближение.
class ConstructiveReal {
private:
    std::function<double(double)> generator;

public:
    // Конструктор от генератора (вычислимой функции)
    ConstructiveReal(std::function<double(double)> gen) : generator(gen) {}

    // Конструктор от константы
    ConstructiveReal(double value) {
        generator = [value](double eps) { return value; };
    }

    // Вычисление с заданной точностью
    double evaluate(double epsilon = 1e-7) const {
        return generator(epsilon);
    }

    // Арифметические операции над конструктивными числами (ленивое вычисление)
    ConstructiveReal operator+(const ConstructiveReal& other) const {
        auto genA = this->generator;
        auto genB = other.generator;
        return ConstructiveReal([genA, genB](double eps) {
            return genA(eps / 2.0) + genB(eps / 2.0);
        });
    }

    ConstructiveReal operator*(const ConstructiveReal& other) const {
        auto genA = this->generator;
        auto genB = other.generator;
        return ConstructiveReal([genA, genB](double eps) {
            // Упрощенная оценка для умножения
            return genA(eps) * genB(eps);
        });
    }
};
