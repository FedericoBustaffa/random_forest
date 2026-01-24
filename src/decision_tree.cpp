#include "decision_tree.hpp"

#include <numeric>
#include <random>

#include "counter.hpp"
#include "tree_functions.hpp"
#include "utils.hpp"

DecisionTree::DecisionTree(size_t max_depth, bool bootstrap,
                           int64_t random_state)
    : m_MaxDepth(max_depth), m_Bootstrap(bootstrap), m_RandomState(random_state)
{
    if (random_state == -1)
    {
        std::random_device rd;
        random_state = rd();
    }
}

int64_t DecisionTree::grow(const std::vector<std::vector<float>>& X,
                           const std::vector<uint8_t>& y,
                           std::vector<size_t>& indices, size_t n_labels,
                           size_t depth)
{
    if (m_MaxDepth != 0 && depth == m_MaxDepth)
    {
        m_Tree.emplace_back(majority(y, indices));
        return m_Tree.size() - 1;
    }

    if (indices.empty())
        return -1;

    size_t n_features = X.size();
    float best_threshold = 0;
    float best_gain = -1;
    size_t best_feature = 0;

    // compute only once
    float parent_entropy = entropy(y, indices);

    for (size_t i = 0; i < n_features; i++)
    {
        // order with indices
        std::vector<size_t> order = argsort(X[i], indices);

        Counter left_counters(n_labels), right_counters(n_labels);
        for (size_t j = 0; j < indices.size(); j++)
            right_counters[y[indices[order[j]]]]++;

        // candidate thresholds
        float prev_label = y[indices[order[0]]];
        for (size_t j = 1; j < indices.size(); j++)
        {
            uint8_t label = y[indices[order[j - 1]]];
            left_counters[label]++;
            right_counters[label]--;

            float prev_feature = X[i][indices[order[j - 1]]];
            float curr_feature = X[i][indices[order[j]]];

            if (prev_feature == curr_feature)
                continue;

            uint8_t curr_label = y[indices[order[j]]];
            if (prev_label != curr_label)
            {
                float threshold = (prev_feature + curr_feature) * 0.5;
                float gain = informationGain(parent_entropy, left_counters,
                                             right_counters);
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

    if (best_gain <= 1e-6)
    {
        m_Tree.emplace_back(majority(y, indices));
        return m_Tree.size() - 1;
    }

    std::vector<size_t> left;
    std::vector<size_t> right;
    left.reserve(indices.size());
    right.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); i++)
    {
        if (X[best_feature][indices[i]] <= best_threshold)
            left.push_back(indices[i]);
        else
            right.push_back(indices[i]);
    }
    indices.clear();

    int64_t idx = m_Tree.size();
    m_Tree.emplace_back(best_feature, best_threshold);
    m_Tree[idx].left = grow(X, y, left, n_labels, depth + 1);
    m_Tree[idx].right = grow(X, y, right, n_labels, depth + 1);

    return idx;
}

void DecisionTree::fit(const std::vector<std::vector<float>>& X,
                       const std::vector<uint8_t>& y)
{
    std::vector<size_t> indices;
    if (m_Bootstrap)
        indices = bootstrap(y.size(), m_RandomState);
    else
    {
        indices.resize(y.size());
        std::iota(indices.begin(), indices.end(), 0);
    }

    size_t n_labels = count_labels(y, indices);

    grow(transpose(X), y, indices, n_labels, 1);
    m_Tree.shrink_to_fit();
}

uint8_t DecisionTree::predict_one(const std::vector<float>& x, int64_t i)
{
    const Node& node = m_Tree[i];
    if (node.label != -1)
        return node.label;

    if (x[node.feature] < node.threshold)
        return predict_one(x, node.left);
    else
        return predict_one(x, node.right);
}

std::vector<uint8_t> DecisionTree::predict(
    const std::vector<std::vector<float>>& X)
{
    std::vector<uint8_t> labels(X.size());
    for (size_t i = 0; i < X.size(); i++)
        labels[i] = predict_one(X[i], 0);

    return labels;
}

size_t DecisionTree::compute_depth(int64_t i) const
{
    if (i == -1)
        return 0;

    const Node& node = m_Tree[i];
    return 1 + std::max(compute_depth(node.left), compute_depth(node.right));
}

DecisionTree::~DecisionTree() {}
