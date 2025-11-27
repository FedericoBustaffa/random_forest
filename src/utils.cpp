#include "utils.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "view.hpp"

std::vector<size_t> argsort(const View<double>& v)
{
    std::vector<size_t> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);

    auto compare = [&v](const auto& a, const auto& b) { return v[a] < v[b]; };
    std::sort(indices.begin(), indices.end(), compare);

    return indices;
}

std::unordered_map<uint32_t, size_t> count(const View<uint32_t>& y)
{
    std::unordered_map<uint32_t, size_t> counter;
    for (size_t i = 0; i < y.size(); i++)
        counter[y[i]]++;

    return counter;
}

uint32_t majority(const View<uint32_t>& y)
{
    std::unordered_map<uint32_t, size_t> counter = count(y);
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

std::vector<size_t> bootstrap(size_t n_samples)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(0, n_samples - 1);

    std::vector<size_t> indices(n_samples);
    for (size_t i = 0; i < n_samples; i++)
    {
        indices[i] = dist(rng);
    }

    return indices;
}

double accuracy(const std::vector<unsigned int>& predictions,
                const std::vector<unsigned int>& correct)
{
    double counter = 0.0;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == correct[i])
            counter++;
    }

    return counter / predictions.size();
}
