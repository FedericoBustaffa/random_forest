#include "decision_tree.hpp"

#include "tree_functions.hpp"
#include "utils.hpp"

DecisionTree::DecisionTree(size_t max_depth) : m_MaxDepth(max_depth) {}

void DecisionTree::grow(const std::vector<std::vector<double>>& X,
                        const std::vector<uint32_t>& y,
                        std::vector<size_t> indices, size_t depth)
{
    if (m_MaxDepth != 0 && depth == m_MaxDepth)
    {
        m_Tree.emplace_back(majority(y, indices));
        return;
    }

    if (indices.empty())
        return;

    double parent_entropy = entropy(count(y, indices));
    if (parent_entropy == 0.0)
    {
        m_Tree.emplace_back(y[indices[0]]);
        return;
    }

    size_t n_features = X.size();
    double best_threshold = 0;
    double best_gain = -1;
    size_t best_feature = 0;

    for (size_t i = 0; i < n_features; i++)
    {
        // order with indices
        std::vector<size_t> order = argsort(X[i], indices);

        std::unordered_map<uint32_t, size_t> left_counters;
        std::unordered_map<uint32_t, size_t> right_counters;
        for (size_t j = 0; j < indices.size(); j++)
            right_counters[y[indices[order[j]]]]++;

        // candidate thresholds
        double prev_label = y[indices[order[0]]];
        for (size_t j = 1; j < indices.size(); j++)
        {
            uint32_t label = y[indices[order[j - 1]]];
            left_counters[label]++;
            right_counters[label]--;

            double prev_feature = X[i][indices[order[j - 1]]];
            double curr_feature = X[i][indices[order[j]]];

            if (prev_feature == curr_feature)
                continue;

            uint32_t curr_label = y[indices[order[j]]];
            if (prev_label != curr_label)
            {
                double threshold = (prev_feature + curr_feature) / 2.0;
                double gain = informationGain(left_counters, right_counters);
                gain = parent_entropy - gain;

                if (gain > best_gain)
                {
                    best_gain = gain;
                    best_threshold = threshold;
                    best_feature = i;
                }
                prev_label = curr_label;
            }
        }
    }

    if (best_gain <= 1e-8)
    {
        m_Tree.push_back(majority(y, indices));
        return;
    }

    std::vector<size_t> left;
    std::vector<size_t> right;

    for (size_t i = 0; i < indices.size(); i++)
    {
        if (X[best_feature][indices[i]] <= best_threshold)
            left.push_back(indices[i]);
        else
            right.push_back(indices[i]);
    }

    m_Tree.emplace_back(best_feature, best_threshold);
    grow(X, y, left, depth + 1);
    grow(X, y, right, depth + 1);
}

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<uint32_t>& y,
                       const std::vector<size_t>& indices)
{
    grow(X, y, indices, 1);
}

uint32_t DecisionTree::predict_one(int64_t idx, const std::vector<double>& x)
{
    const Node& node = m_Tree[idx];
    if (node.label != -1)
        return node.label;

    if (x[node.feature] < node.threshold)
        return predict_one(node.left, x);
    else
        return predict_one(node.right, x);
}

std::vector<uint32_t> DecisionTree::predict(
    const std::vector<std::vector<double>>& X)
{
    std::vector<uint32_t> labels(X.size());
    for (size_t i = 0; i < X.size(); i++)
        labels[i] = predict_one(0, X[i]);

    return labels;
}

size_t DecisionTree::compute_depth(int64_t idx) const
{
    if (idx == -1)
        return 0;

    const Node& node = m_Tree[idx];
    return 1 + std::max(compute_depth(node.left), compute_depth(node.right));
}

size_t DecisionTree::depth() const { return compute_depth(0); }

DecisionTree::~DecisionTree() {}
