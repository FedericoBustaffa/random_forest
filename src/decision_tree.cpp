#include "decision_tree.hpp"

#include <cmath>
#include <unordered_map>

#include "utils.hpp"

DecisionTree::DecisionTree() {}

double DecisionTree::entropy(const TensorView& y)
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

double DecisionTree::information_gain(const TensorView& X, size_t feature,
                                      double threshold)
{
    return 0.0;
}

void DecisionTree::fit(const TensorView& X, const TensorView& y)
{
    size_t n_features = X.shape()[1];
    double best_threshold, best_gain;
    size_t best_feature;

    for (size_t i = 0; i < n_features; i++)
    {
        // sort by feature
        std::vector<size_t> indices = argsort(X(i, 1));

        // candidate thresholds
        double current_class = X[indices[0]];
        double threshold, gain;
        for (size_t j = 1; j < indices.size(); j++)
        {
            if (current_class != X[indices[j]])
            {
                threshold = (X[indices[j - 1]] + X[indices[j]]) / 2.0;

                gain = information_gain(X, i, threshold);
                if (gain > best_gain)
                {
                    best_gain = gain;
                    best_threshold = threshold;
                    best_feature = i;
                }
            }
        }
    }
}

// Tensor DecisionTree::predict(const Tensor& X) { return {}; }

DecisionTree::~DecisionTree() {}
