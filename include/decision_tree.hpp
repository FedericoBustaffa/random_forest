#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <vector>

class decision_tree
{
public:
    decision_tree();

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);

    std::vector<double> predict(const std::vector<std::vector<double>>& X);

    ~decision_tree();

private:
    double entropy(const std::vector<double>& y);

private:
};

#endif
