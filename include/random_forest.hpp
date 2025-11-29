#ifndef RANDOM_FOREST_HPP
#define RANDOM_FOREST_HPP

#include <cstdint>
#include <vector>

#include "decision_tree.hpp"

enum class Policy
{
    Sequential,
    OpenMP,
    Invalid
};

class RandomForest
{
public:
    RandomForest(size_t estimators, size_t max_depth = 0,
                 Policy policy = Policy::Sequential);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<uint32_t> y);

    std::vector<uint32_t> predict(const std::vector<std::vector<double>>& X);

    std::vector<size_t> depths() const;

    ~RandomForest();

private:
    void seq_fit(const std::vector<std::vector<double>>& X,
                 const std::vector<uint32_t> y);

    void omp_fit(const std::vector<std::vector<double>>& X,
                 const std::vector<uint32_t> y);

    std::vector<uint32_t> seq_predict(
        const std::vector<std::vector<double>>& X);

    std::vector<uint32_t> omp_predict(
        const std::vector<std::vector<double>>& X);

private:
    std::vector<DecisionTree> m_Trees;
    Policy m_Policy;
};

#endif
