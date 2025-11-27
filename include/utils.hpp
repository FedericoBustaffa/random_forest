#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "view.hpp"

std::vector<size_t> argsort(const View<double>& v);

std::unordered_map<uint32_t, size_t> count(const View<uint32_t>& y);

uint32_t majority(const View<uint32_t>& y);

std::pair<std::vector<std::vector<double>>, std::vector<uint32_t>> bootstrap(
    const std::vector<std::vector<double>>& X, const std::vector<uint32_t>& y);

double accuracy(const std::vector<unsigned int>& predictions,
                const std::vector<unsigned int>& correct);

#endif
