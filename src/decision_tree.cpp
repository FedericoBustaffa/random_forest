#include "decision_tree.hpp"

#include <numeric>
#include <random>

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

DecisionTree::Node* DecisionTree::grow(
    Node* root, const std::vector<std::vector<double>>& X,
    const std::vector<uint32_t>& y, const std::vector<size_t>& indices,
    size_t depth)
{
    if (m_MaxDepth != 0 && depth == m_MaxDepth)
        return new Node(majority(y, indices));

    size_t n_features = X.size();
    double best_threshold = 0;
    double best_gain = -1;
    size_t best_feature = 0;

    // compute only once
    double parent_entropy = entropy(count(y, indices));

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
        return new Node(majority(y, indices));

    std::vector<size_t> left;
    std::vector<size_t> right;

    for (size_t i = 0; i < indices.size(); i++)
    {
        if (X[best_feature][indices[i]] <= best_threshold)
            left.push_back(indices[i]);
        else
            right.push_back(indices[i]);
    }

    Node* node = new Node(best_feature, best_threshold);
    node->left = grow(node->left, X, y, left, depth + 1);
    node->right = grow(node->right, X, y, right, depth + 1);

    return node;
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
    m_Root = grow(m_Root, T, y, indices, 1);
}

uint32_t DecisionTree::predict_one(Node* node, const std::vector<double>& x)
{
    if (node->label != -1)
        return node->label;

    if (x[node->feature] < node->threshold)
        return predict_one(node->left, x);
    else
        return predict_one(node->right, x);
}

std::vector<uint32_t> DecisionTree::predict(
    const std::vector<std::vector<double>>& X)
{
    std::vector<uint32_t> labels(X.size());
    for (size_t i = 0; i < X.size(); i++)
        labels[i] = predict_one(m_Root, X[i]);

    return labels;
}

size_t DecisionTree::compute_size(Node* node) const
{
    if (node == nullptr)
        return 0;

    return 1 + compute_size(node->left) + compute_size(node->right);
}

size_t DecisionTree::compute_depth(Node* node) const
{
    if (node == nullptr)
        return 0;

    return 1 + std::max(compute_depth(node->left), compute_depth(node->right));
}

void DecisionTree::deallocate(Node* node)
{
    if (node == nullptr)
        return;

    deallocate(node->left);
    deallocate(node->right);

    delete node;
}

DecisionTree::~DecisionTree() { deallocate(m_Root); }
