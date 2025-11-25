#include "decision_tree.hpp"

#include <cmath>
#include <iostream>
#include <unordered_map>

#include "mask.hpp"
#include "utils.hpp"

DecisionTree::DecisionTree() {}

double DecisionTree::entropy(const View<double>& y)
{
    std::unordered_map<unsigned int, double> counters;
    for (size_t i = 0; i < y.size(); i++)
        counters[(unsigned int)y[i]] += 1.0;

    double e = 0.0;
    for (auto& i : counters)
    {
        i.second = i.second / y.size();
        e += -i.second * std::log2(i.second);
    }

    return e;
}

double DecisionTree::informationGain(const View<double>& x,
                                     const View<double>& y, double threshold)
{
    Mask mask = y < threshold;
    double e1 = entropy(y[mask]);
    double e2 = entropy(y[!mask]);
    double gain = entropy(y);

    double ratio = e1 == 0 ? 0.0 : (double)y[mask].size() / y.size();
    gain -= ratio * e1;

    ratio = e2 == 0 ? 0.0 : (double)y[!mask].size() / y.size();
    gain -= ratio * e2;

    return gain;
}

void DecisionTree::grow(const std::vector<View<double>>& X,
                        const View<double>& y)
{
    if (y.size() == 0)
        return;

    if (entropy(y) == 0.0)
        return;

    size_t n_features = X.size();
    double best_threshold = 0;
    double best_gain = -1;
    size_t best_feature = 0;

    for (size_t i = 0; i < n_features; i++)
    {
        // order with indices
        std::vector<size_t> indices = argsort(X[i]);
        View<double> X_sort = X[i][indices];
        View<double> y_sort = y[indices];

        // candidate thresholds
        double current_class = y_sort[0];
        for (size_t j = 1; j < y_sort.size(); j++)
        {
            if (current_class != y_sort[j])
            {
                double threshold = (X_sort[j - 1] + X_sort[j]) / 2.0;
                double gain = informationGain(X_sort, y_sort, threshold);
                if (gain > best_gain)
                {
                    best_gain = gain;
                    best_threshold = threshold;
                    best_feature = i;
                }
                current_class = y_sort[j];
            }
        }
    }

    Mask mask = X[best_feature] < best_threshold;
    std::vector<View<double>> X_left;
    std::vector<View<double>> X_right;
    for (size_t i = 0; i < X.size(); i++)
    {
        X_left.push_back(X[i][mask]);
        X_right.push_back(X[i][!mask]);
    }
    std::cout << entropy(y[mask]) << std::endl;
    std::cout << "not " << entropy(y[!mask]) << std::endl;

    grow(X_left, y[mask]);
    grow(X_right, y[!mask]);
}

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y)
{
    std::vector<View<double>> features;
    for (size_t i = 0; i < X.size(); i++)
        features.push_back(X[i]);

    grow(features, y);
}

DecisionTree::~DecisionTree() {}
