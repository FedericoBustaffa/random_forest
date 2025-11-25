#include "decision_tree.hpp"

#include <cmath>
#include <iostream>
#include <unordered_map>

#include "mask.hpp"
#include "utils.hpp"

DecisionTree::DecisionTree() {}

double DecisionTree::entropy(const VectorView& y)
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

double DecisionTree::informationGain(const VectorView& x, const VectorView& y,
                                     double threshold)
{
    Mask mask = x <= threshold;

    double e1 = entropy(y[mask]);
    double e2 = entropy(y[!mask]);

    double gain = entropy(y);

    double ratio = e1 == 0 ? 0.0 : (double)y[mask].size() / y.size();
    gain -= ratio * e1;

    ratio = e2 == 0 ? 0.0 : (double)y[!mask].size() / y.size();
    gain -= ratio * e2;

    return gain;
}

DecisionTree::Node* DecisionTree::grow(
    Node* node, const MatrixView& X, const VectorView& y,
    const std::vector<std::vector<size_t>>& indices, Mask& mask)
{
    if (y[mask].size() == 0)
        return nullptr;

    if (entropy(y[mask]) == 0.0)
        return new Node(0, 0, y[mask][0]);

    size_t n_features = X.cols();
    double best_threshold = 0;
    double best_gain = 0;
    size_t best_feature = 0;

    for (size_t i = 0; i < n_features; i++)
    {
        // sort by feature
        MatrixView X_sort = X[indices[i]][mask];
        VectorView y_sort = y[indices[i]][mask];

        // candidate thresholds
        double current_class = y_sort[0];
        for (size_t j = 1; j < y_sort.size(); j++)
        {
            if (current_class != y_sort[j])
            {
                double threshold = (X_sort(i)[j - 1] + X_sort(i)[j]) / 2.0;
                double gain = informationGain(X_sort(i), y_sort, threshold);
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

    std::cout << best_gain << std::endl;
    node = new Node(best_feature, best_threshold);

    Mask left_mask = mask & (X(best_feature) <= best_threshold);
    node->m_Left = grow(node->m_Left, X, y, indices, left_mask);

    Mask right_mask = mask & !(X(best_feature) <= best_threshold);
    node->m_Right = grow(node->m_Right, X, y, indices, right_mask);

    return node;
}

void DecisionTree::fit(const MatrixView& X, const VectorView& y)
{

    std::vector<std::vector<size_t>> indices;
    for (size_t i = 0; i < X.cols(); i++)
        indices.push_back(argsort(X(i)));

    Mask mask(true, X.rows());

    m_Root = grow(m_Root, X, y, indices, mask);
}

double DecisionTree::visit(Node* node, VectorView x)
{
    if (node->m_Left == nullptr && node->m_Right == nullptr)
        return node->m_Label;

    if (x[node->m_Feature] <= node->m_Threshold)
        return visit(node->m_Left, x);
    else
        return visit(node->m_Right, x);
}

Vector DecisionTree::predict(const MatrixView& X)
{
    std::vector<double> y;
    for (size_t i = 0; i < X.rows(); i++)
        y.push_back(visit(m_Root, X[i]));

    return Vector(y.data(), y.size());
}

void DecisionTree::deallocate(Node* node)
{
    if (node == nullptr)
        return;

    deallocate(node->m_Left);
    deallocate(node->m_Right);

    delete node;
}

DecisionTree::~DecisionTree() { deallocate(m_Root); }
