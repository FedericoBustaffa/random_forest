#ifndef TREE_FUNCTIONS_HPP
#define TREE_FUNCTIONS_HPP

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "view.hpp"

double entropy(const std::unordered_map<uint32_t, size_t>& counters);

double entropy(const View<uint32_t>& y);

double informationGain(double parent_entropy,
                       const std::unordered_map<uint32_t, size_t>& left,
                       const std::unordered_map<uint32_t, size_t>& right);

#endif
