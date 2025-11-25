#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <vector>

#include "view.hpp"

class DecisionTree
{
public:
    DecisionTree();

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);

    ~DecisionTree();

private:
    double entropy(const View<double>& y);

    double informationGain(const View<double>& feature, const View<double>& y,
                           double threshold);

    void grow(const std::vector<View<double>>& X, const View<double>& y);

private:
};

#endif
