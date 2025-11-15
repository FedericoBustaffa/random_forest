#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include "tensor.hpp"

class DecisionTree
{
public:
    DecisionTree();

    void fit(const TensorView& X, const TensorView& y);

    // Tensor predict(const Tensor& X);

    ~DecisionTree();

private:
    double entropy(const TensorView& y);

    double information_gain(const TensorView& X, size_t feature,
                            double threshold);

private:
};

#endif
