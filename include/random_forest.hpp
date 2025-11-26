#ifndef RANDOM_FOREST_HPP
#define RANDOM_FOREST_HPP

#include <cstdint>
#include <vector>

#include "decision_tree.hpp"

class RandomForest
{
public:
    RandomForest(size_t estimators);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<uint32_t> y);

    std::vector<uint32_t> predict(const std::vector<std::vector<double>>& X);

    std::vector<size_t> depths() const;

    ~RandomForest();

private:
    std::vector<std::vector<double>> bootstrap(
        const std::vector<std::vector<double>>& X) const;

private:
    std::vector<DecisionTree> m_Trees;
};

#endif
