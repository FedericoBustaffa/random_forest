#include "random_forest.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

void RandomForest::seq_fit(const std::vector<std::vector<double>>& X,
                           const std::vector<uint32_t>& y)
{
    for (size_t i = 0; i < m_Trees.size(); i++)
        m_Trees[i].fit(X, y);
}

std::vector<uint32_t> RandomForest::seq_predict(
    const std::vector<std::vector<double>>& X)
{
    std::vector<std::unordered_map<uint32_t, size_t>> counters(X.size());
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
