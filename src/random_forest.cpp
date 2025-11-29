#include "random_forest.hpp"

#include <cstdio>

RandomForest::RandomForest(size_t estimators, size_t max_depth, Policy policy)
    : m_Trees(estimators, max_depth), m_Policy(policy)
{
}

void RandomForest::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<uint32_t> y)
{
    switch (m_Policy)
    {
    case Policy::Sequential:
        seq_fit(X, y);
        break;

    case Policy::OpenMP:
        omp_fit(X, y);
        break;

    case Policy::Invalid:
        break;
    }
}

std::vector<uint32_t> RandomForest::predict(
    const std::vector<std::vector<double>>& X)
{
    switch (m_Policy)
    {
    case Policy::Sequential:
        return seq_predict(X);

    case Policy::OpenMP:
        return omp_predict(X);

    default:
        return {};
    }
}

std::vector<size_t> RandomForest::depths() const
{
    std::vector<size_t> out(m_Trees.size());
    for (size_t i = 0; i < m_Trees.size(); i++)
        out[i] = m_Trees[i].depth();

    return out;
}

RandomForest::~RandomForest() {}
