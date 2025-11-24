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

DecisionTree::Node* DecisionTree::grow(Node* node,
                                       const std::vector<VectorView>& X,
                                       const VectorView& y)
{
    if (y.size() == 0)
        return nullptr;

    if (entropy(y) == 0.0)
        return new Node(0, 0, y[0]);

    size_t n_features = X.size();
    double best_threshold = 0;
    double best_gain = -1;
    size_t best_feature = 0;

    for (size_t i = 0; i < n_features; i++)
    {
        std::cout << "current feature: " << i << std::endl;
        // sort by feature
        std::vector<size_t> indices = argsort(X[i]);
        VectorView X_sort = X[i][indices];
        VectorView y_sort = y[indices];

        // candidate thresholds
        double current_class = y_sort[0];
        for (size_t j = 1; j < y_sort.size(); j++)
        {
            std::cout << j << std::endl;
            if (current_class != y_sort[j])
            {
                double threshold = (X_sort[j - 1] + X_sort[j]) / 2.0;
                double gain = informationGain(X_sort, y_sort, threshold);
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

    node = new Node(best_feature, best_threshold);
    Mask mask = X[best_feature] <= best_threshold;

    std::cout << "best feature: " << best_feature << std::endl;
    std::cout << "best threshold: " << best_threshold << std::endl;

    std::vector<VectorView> X_left, X_right;
    for (size_t i = 0; i < X.size(); i++)
    {
        X_left.emplace_back(X[i][mask]);
        X_right.emplace_back(X[i][!mask]);
    }

    node->m_Left = grow(node->m_Left, X_left, y[mask]);
    node->m_Right = grow(node->m_Right, X_right, y[!mask]);

    return node;
}

void DecisionTree::fit(const MatrixView& X, const VectorView& y)
{
    std::vector<VectorView> features;
    for (size_t i = 0; i < X.cols(); i++)
        features.push_back(X(i).copy());

    m_Root = grow(m_Root, features, y);
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
