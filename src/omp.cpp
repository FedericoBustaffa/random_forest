#include "random_forest.hpp"

#include "utils.hpp"

void RandomForest::omp_fit(const std::vector<std::vector<double>>& X,
                           const std::vector<uint32_t> y)
{
    auto T = transpose(X);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < m_Trees.size(); i++)
    {
        std::vector<size_t> indices = bootstrap(T[0].size());
        m_Trees[i].fit(T, y, indices);
    }
}

std::vector<uint32_t> RandomForest::omp_predict(
    const std::vector<std::vector<double>>& X)
{
    // predict the same batch in parallel
    std::vector<std::vector<uint32_t>> y(m_Trees.size());
#pragma omp parallel for
    for (size_t i = 0; i < m_Trees.size(); i++)
        y[i] = m_Trees[i].predict(X);

    // count votes
    std::vector<std::unordered_map<uint32_t, size_t>> counters(y[0].size());
#pragma omp parallel for
    for (size_t i = 0; i < counters.size(); i++)
    {
        for (size_t j = 0; j < y.size(); j++)
        {
            const std::vector<uint32_t>& pred = y[j];
            counters[i][pred[i]]++;
        }
    }

    // compute majority
    std::vector<uint32_t> prediction(counters.size());
#pragma omp parallel for
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
        prediction[i] = value;
    }

    return prediction;
}
