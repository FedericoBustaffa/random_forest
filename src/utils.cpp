#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>

std::vector<size_t> argsort(const std::vector<double>& v,
                            const std::vector<size_t>& indices)
{
    std::vector<size_t> order(indices.size());
    std::iota(order.begin(), order.end(), 0);

    auto compare = [&](const auto& a, const auto& b) {
        return v[indices[a]] < v[indices[b]];
    };
    std::sort(order.begin(), order.end(), compare);

    return order;
}

std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>>& X)
{
    size_t rows = X.size();
    size_t cols = X[0].size();

    std::vector<std::vector<double>> T(X[0].size());
    for (size_t i = 0; i < cols; i++)
        T[i].reserve(rows);

    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            T[j].push_back(X[i][j]);

    return T;
}

std::unordered_map<uint32_t, size_t> count(const std::vector<uint32_t>& y,
                                           const std::vector<size_t>& indices)
{
    std::unordered_map<uint32_t, size_t> counter;
    for (size_t i = 0; i < indices.size(); i++)
        counter[y[indices[i]]]++;

    return counter;
}

uint32_t majority(const std::vector<uint32_t>& y,
                  const std::vector<size_t>& indices)
{
    std::unordered_map<uint32_t, size_t> counter = count(y, indices);
    uint32_t value = 0;
    size_t best_counter = 0;
    for (const auto& kv : counter)
    {
        if (kv.second > best_counter)
        {
            best_counter = kv.second;
            value = kv.first;
        }
    }

    return value;
}

std::pair<std::vector<size_t>, std::vector<size_t>> train_test_split(
    size_t n_samples, float test_size, int seed)
{
    std::random_device rd;
    if (seed < 0)
        seed = rd();

    std::mt19937 engine(seed);

    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), engine);

    size_t n_test = static_cast<size_t>(n_samples * test_size);
    std::vector<size_t> train_indices(indices.begin() + n_test, indices.end());
    std::vector<size_t> test_indices(indices.begin(), indices.begin() + n_test);

    return {train_indices, test_indices};
}

std::vector<size_t> bootstrap(size_t n_samples, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, n_samples - 1);

    std::vector<size_t> indices(n_samples);
    for (size_t i = 0; i < n_samples; i++)
        indices[i] = dist(rng);

    return indices;
}

Backend to_backend(const std::string& s)
{
    if (s == "seq")
        return Backend::Sequential;

    if (s == "omp")
        return Backend::OpenMP;

    if (s == "ff")
        return Backend::FastFlow;

    if (s == "mpi")
        return Backend::MPI;

    return Backend::Invalid;
}
