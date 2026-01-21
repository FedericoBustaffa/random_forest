#include "random_forest.hpp"

#include <cstddef>
#include <cstdint>

#include "counter.hpp"

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
    std::vector<Counter> counters(X.size(), m_Labels);
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
        for (size_t j = 0; j < counters[i].size(); ++j)
        {
            if (counters[i][j] > counter)
            {
                counter = counters[i][j];
                value = j;
            }
        }
        prediction[i] = value;
    }

    return prediction;
}
