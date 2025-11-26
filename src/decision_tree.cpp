#include "decision_tree.hpp"

#include <cmath>

#include "mask.hpp"
#include "utils.hpp"

DecisionTree::DecisionTree() {}

double DecisionTree::entropy(const View<uint32_t>& y)
{
    std::unordered_map<size_t, size_t> counters;
    for (size_t i = 0; i < y.size(); i++)
        counters[y[i]] += 1;

    double e = 0.0;
    double proportion;
    for (auto& i : counters)
    {
        proportion = (double)i.second / y.size();
        e += -proportion * std::log2(proportion);
    }

    return e;
}

double DecisionTree::entropy(const std::unordered_map<size_t, size_t>& counters,
                             size_t size)
{
    double e = 0.0;
    double proportion;
    for (auto& i : counters)
    {
        if (i.second == 0)
            continue;

        proportion = (double)i.second / size;
        e += -proportion * std::log2(proportion);
    }

    return e;
}

double DecisionTree::informationGain(const View<double>& x,
                                     const View<uint32_t>& y, double threshold)
{
    Mask mask = x < threshold;

    View y_left = y[mask];
    View y_right = y[!mask];

    if (y_left.size() == 0 || y_right.size() == 0)
        return 0.0;

    double e_left = entropy(y_left);
    double e_right = entropy(y_right);

    double ratio_left = (double)y_left.size() / y.size();
    double ratio_right = (double)y_right.size() / y.size();

    return (ratio_left * e_left) + (ratio_right * e_right);
}

double DecisionTree::informationGain(
    const std::array<std::unordered_map<size_t, size_t>, 2> counters)
{
    size_t left_size = 0;
    for (const auto& i : counters[0])
        left_size += i.second;

    size_t right_size = 0;
    for (const auto& i : counters[1])
        right_size += i.second;

    if (left_size == 0 || right_size == 0)
        return 0.0;

    size_t size = left_size + right_size;

    double e_left = entropy(counters[0], left_size);
    double e_right = entropy(counters[1], right_size);

    double ratio_left = (double)left_size / size;
    double ratio_right = (double)right_size / size;

    return (ratio_left * e_left) + (ratio_right * e_right);
}

DecisionTree::Node* DecisionTree::grow(Node* root,
                                       const std::vector<View<double>>& X,
                                       const View<uint32_t>& y)
{
    if (y.size() == 0)
        return nullptr;

    double parent_entropy = entropy(y);
    if (parent_entropy == 0.0)
        return new Node(y[0]);

    size_t n_features = X.size();
    double best_threshold = 0;
    double best_gain = -1;
    size_t best_feature = 0;

    for (size_t i = 0; i < n_features; i++)
    {
        // order with indices
        std::vector<size_t> order = argsort(X[i]);
        const View<double>& X_sort = X[i][order];
        const View<uint32_t>& y_sort = y[order];

        std::array<std::unordered_map<size_t, size_t>, 2> counters;
        for (size_t j = 0; j < y_sort.size(); j++)
            counters[1][y_sort[j]]++;

        // candidate thresholds
        double current_class = y_sort[0];
        for (size_t j = 1; j < y_sort.size(); j++)
        {
            counters[0][y_sort[j - 1]]++;
            counters[1][y_sort[j - 1]]--;

            if (X_sort[j - 1] == X_sort[j])
                continue;

            if (current_class != y_sort[j])
            {
                double threshold = (X_sort[j - 1] + X_sort[j]) / 2.0;
                double gain = informationGain(counters);
                gain = parent_entropy - gain;

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

    if (best_gain <= 1e-6)
    {
        std::unordered_map<size_t, size_t> final_counters;
        for (size_t k = 0; k < y.size(); k++)
            final_counters[y[k]] += 1;

        size_t majority_class = 0;
        size_t max_count = 0;
        for (const auto& pair : final_counters)
        {
            if (pair.second > max_count)
            {
                max_count = pair.second;
                majority_class = pair.first;
            }
        }
        return new Node(majority_class);
    }

    // std::cout << "best threshold: " << best_threshold << std::endl;
    // std::cout << "best feature: " << best_feature << std::endl;

    Mask mask = X[best_feature] < best_threshold;

    std::vector<View<double>> X_left;
    std::vector<View<double>> X_right;

    for (size_t i = 0; i < X.size(); i++)
    {
        X_left.push_back(X[i][mask]);
        X_right.push_back(X[i][!mask]);
    }

    Node* node = new Node(best_feature, best_threshold);
    node->left = grow(node->left, X_left, y[mask]);
    node->right = grow(node->right, X_right, y[!mask]);

    return node;
}

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<uint32_t>& y)
{
    std::vector<std::vector<size_t>> indices;

    std::vector<View<double>> features;
    for (size_t i = 0; i < X.size(); i++)
        features.emplace_back(X[i].data(), X[i].size());

    View targets(y.data(), y.size());

    m_Root = grow(m_Root, features, targets);
}

uint32_t DecisionTree::visit(Node* node, const std::vector<double>& x)
{
    if (node->label != -1)
        return node->label;

    if (x[node->feature] < node->threshold)
        return visit(node->left, x);
    else
        return visit(node->right, x);
}

std::vector<uint32_t> DecisionTree::predict(
    const std::vector<std::vector<double>>& X)
{
    std::vector<double> pattern(X.size());
    std::vector<uint32_t> labels(X[0].size());

    for (size_t i = 0; i < X[0].size(); i++)
    {
        for (size_t j = 0; j < X.size(); j++)
            pattern[j] = X[j][i];

        labels[i] = visit(m_Root, pattern);
    }

    return labels;
}

size_t DecisionTree::compute_depth(Node* node) const
{
    if (node == nullptr)
        return 0;

    return 1 + std::max(compute_depth(node->left), compute_depth(node->right));
}

size_t DecisionTree::depth() const { return compute_depth(m_Root); }

void DecisionTree::deallocate(Node* node)
{
    if (node == nullptr)
        return;

    deallocate(node->left);
    deallocate(node->right);

    delete node;
}

DecisionTree::~DecisionTree() { deallocate(m_Root); }
