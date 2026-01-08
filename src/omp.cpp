#include "random_forest.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

void RandomForest::omp_fit(const std::vector<std::vector<double>>& X,
                           const std::vector<uint32_t>& y)
{
#pragma omp parallel for schedule(dynamic) num_threads(m_Threads)
    for (size_t i = 0; i < m_Trees.size(); i++)
        m_Trees[i].fit(X, y);
}

std::vector<uint32_t> RandomForest::omp_predict(
    const std::vector<std::vector<double>>& X)
{
    // predict the same batch in parallel
    std::vector<std::vector<uint32_t>> y(m_Trees.size());
#pragma omp parallel for num_threads(m_Threads)
    for (size_t i = 0; i < m_Trees.size(); i++)
        y[i] = m_Trees[i].predict(X);

    // count votes and compute majority
    std::vector<std::unordered_map<uint32_t, size_t>> counters(y[0].size());
    std::vector<uint32_t> prediction(counters.size());
#pragma omp parallel for num_threads(m_Threads)
    for (size_t i = 0; i < counters.size(); i++)
    {
        for (size_t j = 0; j < y.size(); j++)
        {
            const std::vector<uint32_t>& pred = y[j];
            counters[i][pred[i]]++;
        }

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
        prediction[i] = value;
    }

    return prediction;
}
