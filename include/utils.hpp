#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "view.hpp"

std::vector<size_t> argsort(const View<double>& v);

std::unordered_map<uint32_t, size_t> count(const View<uint32_t>& y);

uint32_t majority(const View<uint32_t>& y);

std::vector<size_t> bootstrap(size_t n_samples);

double accuracy(const std::vector<unsigned int>& predictions,
                const std::vector<unsigned int>& correct);

#endif
