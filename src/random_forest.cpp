#include "random_forest.hpp"

#include <random>

RandomForest::RandomForest(size_t estimators) : m_Trees(estimators) {}

std::vector<std::vector<double>> RandomForest::bootstrap(
    const std::vector<std::vector<double>>& X) const
{
    size_t n_samples = X[0].size();

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(0, n_samples - 1);

    std::vector<std::vector<double>> out(X.size());
    for (size_t i = 0; i < X.size(); i++)
    {
        out[i].reserve(n_samples);
        for (size_t j = 0; j < n_samples; j++)
            out[i].push_back(X[i][dist(rng)]);
    }

    return out;
}

void RandomForest::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<uint32_t> y)
{
    double total = 0.0;
    for (size_t i = 0; i < m_Trees.size(); i++)
        m_Trees[i].fit(bootstrap(X), y);

    std::printf("no bootstrap time: %.4f seconds\n", total);
}

std::vector<uint32_t> RandomForest::predict(
    const std::vector<std::vector<double>>& X)
{
    std::vector<std::unordered_map<uint32_t, size_t>> counters(m_Trees.size());
    for (size_t i = 0; i < m_Trees.size(); i++)
    {
        std::vector<uint32_t> pred = m_Trees[i].predict(X);
        for (size_t j = 0; j < pred.size(); j++)
            counters[j][pred[j]]++;
    }

    std::vector<uint32_t> prediction;
    for (size_t i = 0; i < counters.size(); i++)
    {
        std::vector<size_t> values;
        for (const auto& kv : counters[i])
            values.push_back(kv.second);
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
