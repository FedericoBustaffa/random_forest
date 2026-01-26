#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>

std::vector<size_t> argsort(const std::vector<float>& v,
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

std::vector<std::vector<float>> transpose(
    const std::vector<std::vector<float>>& X)
{
    size_t rows = X.size();
    size_t cols = X[0].size();

    std::vector<std::vector<float>> T(X[0].size());
    for (size_t i = 0; i < cols; i++)
        T[i].reserve(rows);

    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            T[j].push_back(X[i][j]);

    return T;
}

size_t count_labels(const std::vector<uint8_t>& y)
{
    size_t found = 0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        if (y[i] >= found)
            found = y[i];
    }

    return found + 1;
}

size_t count_labels(const std::vector<uint8_t>& y,
                    const std::vector<size_t>& indices)
{
    size_t found = 0;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (y[indices[i]] >= found)
            found = y[indices[i]];
    }

    return found + 1;
}

Counter count(const std::vector<uint8_t>& y, const std::vector<size_t>& indices)
{
    Counter counter(count_labels(y, indices));
    for (size_t i = 0; i < indices.size(); i++)
        counter[y[indices[i]]]++;

    return counter;
}

uint8_t majority(const std::vector<uint8_t>& y,
                 const std::vector<size_t>& indices)
{
    Counter counter = count(y, indices);
    uint8_t value = 0;
    size_t best_counter = 0;
    for (size_t i = 0; i < counter.size(); ++i)
    {
        if (counter[i] > best_counter)
        {
            best_counter = counter[i];
            value = i;
        }
    }

    return value;
}

DataSplit train_test_split(const std::vector<std::vector<float>>& X,
                           const std::vector<uint8_t>& y, float test_size,
                           int seed)
{
    if (seed < 0)
    {
        std::random_device rd;
        seed = rd();
    }

    std::mt19937 engine(seed);

    size_t n_samples = X.size();
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), engine);

    size_t n_test = static_cast<size_t>(n_samples * test_size);
    std::vector<size_t> train_indices(indices.begin() + n_test, indices.end());
    std::vector<size_t> test_indices(indices.begin(), indices.begin() + n_test);

    std::vector<std::vector<float>> X_train(n_samples - n_test), X_test(n_test);
    std::vector<uint8_t> y_train(n_samples - n_test), y_test(n_test);

    for (size_t i = 0; i < n_samples - n_test; i++)
    {
        X_train[i] = X[train_indices[i]];
        y_train[i] = y[train_indices[i]];
    }

    for (size_t i = 0; i < n_test; i++)
    {
        X_test[i] = X[test_indices[i]];
        y_test[i] = y[test_indices[i]];
    }

    return DataSplit(X_train, X_test, y_train, y_test);
}

std::vector<size_t> bootstrap(size_t n_samples, uint8_t seed)
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
