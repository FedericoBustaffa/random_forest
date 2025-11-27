#include "random_forest.hpp"
#include "utils.hpp"

#include <cstdio>
#include <unordered_map>

RandomForest::RandomForest(size_t estimators, size_t max_depth)
    : m_Trees(estimators, max_depth)
{
}

void RandomForest::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<uint32_t> y)
{
#pragma omp parallel for
    for (size_t i = 0; i < m_Trees.size(); i++)
    {
        std::vector<size_t> indices = bootstrap(X[0].size());
        std::vector<View<double>> Xb;
        Xb.reserve(X.size());
        for (size_t j = 0; j < X.size(); j++)
        {
            Xb.emplace_back(X[j].data(), X[j].size());
            Xb[j] = Xb[j][indices];
        }

        View<uint32_t> yb(y.data(), y.size());

        m_Trees[i].fit(Xb, yb[indices]);
    }
}

std::vector<uint32_t> RandomForest::predict(
    const std::vector<std::vector<double>>& X)
{
    std::vector<std::unordered_map<uint32_t, size_t>> counters(X[0].size());
    for (size_t i = 0; i < m_Trees.size(); i++)
    {
        std::vector<uint32_t> pred = m_Trees[i].predict(X);
        for (size_t j = 0; j < pred.size(); j++)
            counters[j][pred[j]]++;
    }

    std::vector<uint32_t> prediction;
    for (size_t i = 0; i < counters.size(); i++)
    {
        uint32_t value = 0;
        size_t counter = 0;
        for (const auto& kv : counters[i])
        {
            if (kv.second > counter)
            {
                counter = kv.second;
                value = kv.first;
            }
        }

        prediction.push_back(value);
    }

    return prediction;
}

std::vector<size_t> RandomForest::depths() const
{
    std::vector<size_t> out(m_Trees.size());
    for (size_t i = 0; i < m_Trees.size(); i++)
        out[i] = m_Trees[i].depth();

    return out;
}

RandomForest::~RandomForest() {}
