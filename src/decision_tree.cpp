#include "decision_tree.hpp"

#include "mask.hpp"
#include "tree_functions.hpp"
#include "utils.hpp"

DecisionTree::DecisionTree(size_t max_depth) : m_MaxDepth(max_depth) {}

DecisionTree::Node* DecisionTree::grow(Node* root,
                                       const std::vector<View<double>>& X,
                                       const View<uint32_t>& y, size_t depth)
{
    if (m_MaxDepth != 0 && depth == m_MaxDepth)
        return new Node(majority(y));

    if (y.size() == 0)
        return nullptr;

    double parent_entropy = entropy(count(y));
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

        std::unordered_map<uint32_t, size_t> left_counters;
        std::unordered_map<uint32_t, size_t> right_counters;
        for (size_t j = 0; j < y_sort.size(); j++)
            right_counters[y_sort[j]]++;

        // candidate thresholds
        double current_class = y_sort[0];
        for (size_t j = 1; j < y_sort.size(); j++)
        {
            left_counters[y_sort[j - 1]]++;
            right_counters[y_sort[j - 1]]--;

            if (X_sort[j - 1] == X_sort[j])
                continue;

            if (current_class != y_sort[j])
            {
                double threshold = (X_sort[j - 1] + X_sort[j]) / 2.0;
                double gain = informationGain(left_counters, right_counters);
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
        return new Node(majority(y));

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
    node->left = grow(node->left, X_left, y[mask], depth + 1);
    node->right = grow(node->right, X_right, y[!mask], depth + 1);

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

    m_Root = grow(m_Root, features, targets, 1);
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
