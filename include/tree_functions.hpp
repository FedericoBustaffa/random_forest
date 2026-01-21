#ifndef TREE_FUNCTIONS_HPP
#define TREE_FUNCTIONS_HPP

#include <cstdint>

#include "counter.hpp"
#include "view.hpp"

double entropy(const Counter& counters);

double entropy(const View<uint32_t>& y);

double informationGain(double parent_entropy, const Counter& left,
                       const Counter& right);

#endif
