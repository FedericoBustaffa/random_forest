#include "decision_tree.hpp"

#include "tree_functions.hpp"
#include "utils.hpp"

DecisionTree::DecisionTree(size_t max_depth) : m_MaxDepth(max_depth) {}

DecisionTree::Node* DecisionTree::grow(
    Node* root, const std::vector<std::vector<double>>& X,
    const std::vector<uint32_t>& y, std::vector<size_t> indices, size_t depth)
{
    if (m_MaxDepth != 0 && depth == m_MaxDepth)
        return new Node(majority(y, indices));

    if (indices.size() == 0)
        return nullptr;

    double parent_entropy = entropy(count(y, indices));
    if (parent_entropy == 0.0)
        return new Node(y[indices[0]]);

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
        double current_class = y[indices[order[0]]];
        for (size_t j = 1; j < indices.size(); j++)
        {
            left_counters[y[indices[order[j - 1]]]]++;
            right_counters[y[indices[order[j - 1]]]]--;

            if (X[i][indices[order[j - 1]]] == X[i][indices[order[j]]])
                continue;

            if (current_class != y[indices[order[j]]])
            {
                double threshold =
                    (X[i][indices[order[j - 1]]] + X[i][indices[order[j]]]) /
                    2.0;
                double gain = informationGain(left_counters, right_counters);
                gain = parent_entropy - gain;

                if (gain > best_gain)
                {
                    best_gain = gain;
                    best_threshold = threshold;
                    best_feature = i;
                }
                current_class = y[indices[order[j]]];
            }
        }
    }

    if (best_gain <= 1e-8)
        return new Node(majority(y, indices));

    // std::cout << "best threshold: " << best_threshold << std::endl;
    // std::cout << "best feature: " << best_feature << std::endl;

    std::vector<size_t> left;
    std::vector<size_t> right;

    // left.reserve(X[0].size());
    // right.reserve(X[0].size());

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
                       const std::vector<uint32_t>& y,
                       const std::vector<size_t>& indices)
{
    m_Root = grow(m_Root, X, y, indices, 1);
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
    std::vector<double> pattern(X.size());
    std::vector<uint32_t> labels(X[0].size());

    for (size_t i = 0; i < X[0].size(); i++)
    {
        for (size_t j = 0; j < X.size(); j++)
            pattern[j] = X[j][i];

        labels[i] = predict_one(m_Root, pattern);
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
