#ifndef TREE_FUNCTIONS_HPP
#define TREE_FUNCTIONS_HPP

#include <cstdint>

#include "counter.hpp"

float entropy(const Counter& counters);

float entropy(const std::vector<uint8_t>& y,
              const std::vector<size_t>& indices);

float informationGain(float parent_entropy, const Counter& left,
                      const Counter& right);

#endif
