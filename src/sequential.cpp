#include "random_forest.hpp"

#include <cstddef>
#include <cstdint>

#include "counter.hpp"

void RandomForest::seq_fit(const DataSplit& data)
{
    for (size_t i = 0; i < m_Trees.size(); i++)
        m_Trees[i].fit(data);
}

std::vector<uint8_t> RandomForest::seq_predict(
    const std::vector<std::vector<float>>& X)
{
    std::vector<Counter> counters(X.size(), m_Labels);
    for (size_t i = 0; i < m_Trees.size(); i++)
    {
        std::vector<uint8_t> pred = m_Trees[i].predict(X);
        for (size_t j = 0; j < pred.size(); j++)
            counters[j][pred[j]]++;
    }

    std::vector<uint8_t> prediction;
    for (size_t i = 0; i < counters.size(); i++)
    {
        uint8_t value = 0;
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
