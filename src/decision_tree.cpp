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

int64_t DecisionTree::grow(const std::vector<View<double>>& X,
                           const View<uint32_t>& y, size_t n_labels,
                           size_t depth)
{
    if (m_MaxDepth != 0 && depth == m_MaxDepth)
    {
        m_Tree.emplace_back(majority(y));
        return m_Tree.size() - 1;
    }

    if (y.empty())
        return -1;

    size_t n_features = X.size();
    double best_threshold = 0;
    double best_gain = -1;
    size_t best_feature = 0;

    // compute only once
    double parent_entropy = entropy(y);

    for (size_t i = 0; i < n_features; i++)
    {
        // order with indices
        std::vector<size_t> order = argsort(X[i]);

        Counter left_counters(n_labels), right_counters(n_labels);
        for (size_t j = 0; j < y.size(); j++)
            right_counters[y[order[j]]]++;

        // candidate thresholds
        double prev_label = y[order[0]];
        for (size_t j = 1; j < y.size(); j++)
        {
            uint32_t label = y[order[j - 1]];
            left_counters[label]++;
            right_counters[label]--;

            double prev_feature = X[i][order[j - 1]];
            double curr_feature = X[i][order[j]];

            if (prev_feature == curr_feature)
                continue;

            uint32_t curr_label = y[order[j]];
            if (prev_label != curr_label)
            {
                double threshold = (prev_feature + curr_feature) * 0.5;
                double gain = informationGain(parent_entropy, left_counters,
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
        m_Tree.emplace_back(majority(y));
        return m_Tree.size() - 1;
    }

    std::vector<size_t> left;
    std::vector<size_t> right;

    const std::vector<size_t> indices = y.indices();

    for (size_t i = 0; i < y.size(); i++)
    {
        if (X[best_feature][i] <= best_threshold)
            left.push_back(indices[i]);
        else
            right.push_back(indices[i]);
    }

    std::vector<View<double>> X_left;
    X_left.reserve(left.size());
    for (size_t i = 0; i < X.size(); ++i)
        X_left.emplace_back(X[i], left);
    View<uint32_t> y_left(y, left);

    std::vector<View<double>> X_right;
    X_right.reserve(right.size());
    for (size_t i = 0; i < X.size(); ++i)
        X_right.emplace_back(X[i], right);
    View<uint32_t> y_right(y, right);

    int64_t idx = m_Tree.size();
    m_Tree.emplace_back(best_feature, best_threshold);
    m_Tree[idx].left = grow(X_left, y_left, n_labels, depth + 1);
    m_Tree[idx].right = grow(X_right, y_right, n_labels, depth + 1);

    return idx;
}

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<uint32_t>& y)
{
    std::vector<size_t> indices;
    if (m_Bootstrap)
        indices = bootstrap(y.size(), m_RandomState);
    else
    {
        indices.resize(y.size());
        std::iota(indices.begin(), indices.end(), 0);
    }

    auto T = transpose(X);

    std::vector<View<double>> T_view;
    T_view.reserve(T.size());
    for (size_t i = 0; i < T.size(); ++i)
        T_view.emplace_back(T[i], indices);

    View<uint32_t> y_view(y, indices);

    size_t n_labels = count_labels(y_view);

    grow(T_view, y_view, n_labels, 1);
    m_Tree.shrink_to_fit();
}

uint32_t DecisionTree::predict_one(const std::vector<double>& x, int64_t i)
{
    const Node& node = m_Tree[i];
    if (node.label != -1)
        return node.label;

    if (x[node.feature] < node.threshold)
        return predict_one(x, node.left);
    else
        return predict_one(x, node.right);
}

std::vector<uint32_t> DecisionTree::predict(
    const std::vector<std::vector<double>>& X)
{
    std::vector<uint32_t> labels(X.size());
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
