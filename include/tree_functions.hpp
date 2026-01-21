#ifndef TREE_FUNCTIONS_HPP
#define TREE_FUNCTIONS_HPP

#include <cstddef>
#include <cstdint>
#include <unordered_map>

double entropy(const std::unordered_map<uint32_t, size_t>& counters);

double informationGain(double parent_entropy,
                       const std::unordered_map<uint32_t, size_t>& left,
                       const std::unordered_map<uint32_t, size_t>& right);

#endif
