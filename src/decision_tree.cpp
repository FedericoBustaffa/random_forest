#include "decision_tree.hpp"

#include <cmath>
#include <cstdio>
#include <unordered_map>

DecisionTree::DecisionTree() {}

double DecisionTree::entropy(const Vector& y)
{
    std::unordered_map<double, double> counters;
    for (size_t i = 0; i < y.size(); i++)
        counters[y[i]] += 1.0;

    double e = 0.0;
    for (auto& i : counters)
    {
        i.second = i.second / y.size();
        e += -i.second * std::log2(i.second);
    }

    return e;
}

void DecisionTree::fit(const Matrix& X, const Vector& y)
{
    std::printf("entropy: %.3f\n", entropy(y));
}

Vector DecisionTree::predict(const Matrix& X) { return Vector(nullptr, 0); }

DecisionTree::~DecisionTree() {}
