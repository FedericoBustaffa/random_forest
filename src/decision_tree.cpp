#include "decision_tree.hpp"

#include <cmath>

decision_tree::decision_tree() {}

double entropy(const std::vector<double>& y, double cls)
{
    double pos = 0.0;
    for (size_t i = 0; i < y.size(); i++)
    {
        if (y[i] == cls)
            pos += 1;
    }
    pos = pos / y.size();

    return -pos * std::log2(pos) - (1.0 - pos) * std::log2(1.0 - pos);
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
