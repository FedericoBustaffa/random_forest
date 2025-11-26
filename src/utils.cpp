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

std::pair<std::vector<std::vector<double>>, std::vector<uint32_t>> bootstrap(
    const std::vector<std::vector<double>>& X, const std::vector<uint32_t>& y)
{
    size_t n_samples = X[0].size();

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(0, n_samples - 1);

    std::vector<std::vector<double>> X_out(X.size());
    std::vector<uint32_t> y_out;
    y_out.reserve(n_samples);
    for (size_t i = 0; i < n_samples; i++)
    {
        size_t index = dist(rng);
        for (size_t j = 0; j < X.size(); j++)
            X_out[j].push_back(X[j][index]);
        y_out.push_back(y[index]);
    }

    return {X_out, y_out};
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
