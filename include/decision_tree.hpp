#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include "matrix.hpp"
#include "vector.hpp"

class DecisionTree
{
public:
    DecisionTree();

    void fit(const Matrix& X, const Vector& y);

    Vector predict(const Matrix& X);

    ~DecisionTree();

private:
    double entropy(const Vector& y);

private:
};

#endif
