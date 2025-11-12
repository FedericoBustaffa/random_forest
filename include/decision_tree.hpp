#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include "tensor.hpp"

class DecisionTree
{
public:
    DecisionTree();

    void fit(const Tensor& X, const Tensor& y);

    // Tensor predict(const Tensor& X);

    ~DecisionTree();

private:
    double entropy(const Tensor& y);

private:
};

#endif
