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

double DecisionTree::information_gain(const TensorView& feature,
                                      const TensorView& y, double threshold)
{
    double e = entropy(y);

    std::vector<double> s1;
    std::vector<double> s2;
    for (size_t i = 0; i < y.size(); i++)
    {
        if (y[feature[i]] <= threshold)
            s1.push_back(y);
        else
            s2.push_back(y);
    }

    double e1 = entropy(TensorView(s1.data(), {s1.size()}));
    double e2 = entropy(TensorView(s2.data(), {s2.size()}));

    double gain = e;
    gain -= (double)s1.size() / y.size() * e1;
    gain -= (double)s2.size() / y.size() * e2;

    return gain;
}

void DecisionTree::split(size_t feature, double threshold)
{
    Node n(feature, threshold);
    nodes.push_back(n);
}

void DecisionTree::fit(const TensorView& X, const TensorView& y)
{
    size_t n_features = X.shape()[1];
    double best_threshold, best_gain = 0;
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

                gain = information_gain(X(i), y, threshold);
                if (gain > best_gain)
                {
                    best_gain = gain;
                    best_threshold = threshold;
                    best_feature = i;
                }
            }
        }
    }

    // perform a split
    split(best_feature, best_threshold);
}

// Tensor DecisionTree::predict(const Tensor& X) { return ; }

DecisionTree::~DecisionTree() {}
