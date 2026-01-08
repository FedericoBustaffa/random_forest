#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "random_forest.hpp"

std::vector<size_t> argsort(const std::vector<double>& v,
                            const std::vector<size_t>& indices);

std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>>& X);

std::unordered_map<uint32_t, size_t> count(const std::vector<uint32_t>& y,
                                           const std::vector<size_t>& indices);

uint32_t majority(const std::vector<uint32_t>& y,
                  const std::vector<size_t>& indices);

std::pair<std::vector<size_t>, std::vector<size_t>> train_test_split(
    size_t n_samples, float test_size, int seed = -1);

template <typename T>
std::pair<std::vector<T>, std::vector<T>> split(
    const std::vector<T>& v, const std::vector<size_t>& train_idx,
    const std::vector<size_t>& test_idx)
{
    std::vector<T> train(train_idx.size());
    std::vector<T> test(test_idx.size());

    for (size_t i = 0; i < train_idx.size(); i++)
        train[i] = v[train_idx[i]];

    for (size_t i = 0; i < test_idx.size(); i++)
        test[i] = v[test_idx[i]];

    return {train, test};
}

std::vector<size_t> bootstrap(size_t n_samples, uint32_t seed);

Backend to_backend(const std::string& s);

#endif
