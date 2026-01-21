#include "random_forest.hpp"

#include <cstddef>
#include <cstdint>

#include "counter.hpp"

void RandomForest::seq_fit(const std::vector<std::vector<double>>& X,
                           const std::vector<uint32_t>& y)
{
    for (size_t i = 0; i < m_Trees.size(); i++)
        m_Trees[i].fit(X, y);
}

std::vector<uint32_t> RandomForest::seq_predict(
    const std::vector<std::vector<double>>& X)
{
    std::vector<Counter> counters(X.size(), m_Labels);
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
        for (size_t j = 0; j < counters[i].size(); ++j)
        {
            if (counters[i][j] > counter)
            {
                counter = counters[i][j];
                value = j;
            }
        }

        prediction.push_back(value);
    }

    return prediction;
}
