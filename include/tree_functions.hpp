#ifndef TREE_FUNCTIONS_HPP
#define TREE_FUNCTIONS_HPP

#include <cstdint>

#include "counter.hpp"

double entropy(const Counter& counters);

double entropy(const std::vector<uint32_t>& y,
               const std::vector<size_t>& indices);

double informationGain(double parent_entropy, const Counter& left,
                       const Counter& right);

#endif
