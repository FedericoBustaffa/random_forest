#include "random_forest.hpp"

#include "utils.hpp"

void RandomForest::seq_fit(const std::vector<std::vector<double>>& X,
                           const std::vector<uint32_t>& y)
{
    auto T = transpose(X);
    for (size_t i = 0; i < m_Trees.size(); i++)
    {
        uint32_t seed = m_Trees.size() + i;
        std::vector<size_t> indices = bootstrap(T[0].size(), seed);
        m_Trees[i].fit(T, y, indices);
    }
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
