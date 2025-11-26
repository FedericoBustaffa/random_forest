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
    Mask mask = x < threshold;

    View y_left = y[mask];
    View y_right = y[!mask];

    if (y_left.size() == 0 || y_right.size() == 0)
        return 0.0;

    double e_parent = entropy(y);
    double e_left = entropy(y_left);
    double e_right = entropy(y_right);

    double ratio_left = (double)y_left.size() / y.size();
    double ratio_right = (double)y_right.size() / y.size();

    return e_parent - (ratio_left * e_left) - (ratio_right * e_right);
}

void DecisionTree::grow(const std::vector<View<double>>& X,
                        const View<double>& y)
{
    if (y.size() == 0 || entropy(y) == 0.0)
        return;

    size_t n_features = X.size();
    double best_threshold = 0;
    double best_gain = -1;
    size_t best_feature = 0;

    for (size_t i = 0; i < n_features; i++)
    {
        std::cout << "feature: " << i << std::endl;

        // order with indices
        std::vector<size_t> order = argsort(X[i]);
        const View<double>& X_sort = X[i][order];
        const View<double>& y_sort = y[order];

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

    grow(X_left, y[mask]);
    grow(X_right, y[!mask]);
}

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y)
{
    std::vector<std::vector<size_t>> indices;

    std::vector<View<double>> features;
    for (size_t i = 0; i < X.size(); i++)
        features.emplace_back(X[i].data(), X[i].size());

    View targets(y.data(), y.size());

    grow(features, targets);
}

DecisionTree::~DecisionTree() {}
