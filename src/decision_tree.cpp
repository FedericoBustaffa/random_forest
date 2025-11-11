#include "decision_tree.hpp"

#include <cmath>
#include <cstdio>
#include <unordered_map>

decision_tree::decision_tree() {}

double decision_tree::entropy(const std::vector<double>& y)
{
    std::unordered_map<double, double> counters;
    for (size_t i = 0; i < y.size(); i++)
        counters[y[i]] += 1.0;

    double s = 0.0;
    for (auto& i : counters)
    {
        i.second = i.second / y.size();
        s += -i.second * std::log2(i.second);
    }

    return s;
}

void decision_tree::fit(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y)
{
}

std::vector<double> decision_tree::predict(
    const std::vector<std::vector<double>>& X)
{
    return {};
}

decision_tree::~decision_tree() {}
