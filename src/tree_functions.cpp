#include "tree_functions.hpp"

#include <cmath>

#include "utils.hpp"

double entropy(const std::unordered_map<uint32_t, size_t>& counters)
{
    size_t size = 0;
    for (const auto& kv : counters)
        size += kv.second;

    double e = 0.0;
    double proportion;
    for (auto& i : counters)
    {
        if (i.second == 0)
            continue;

        proportion = (double)i.second / size;
        e += -proportion * std::log2(proportion);
    }

    return e;
}

double entropy(const View<uint32_t>& y) { return entropy(count(y)); }

double informationGain(double parent_entropy,
                       const std::unordered_map<uint32_t, size_t>& left,
                       const std::unordered_map<uint32_t, size_t>& right)
{
    size_t left_size = 0;
    for (const auto& i : left)
        left_size += i.second;

    size_t right_size = 0;
    for (const auto& i : right)
        right_size += i.second;

    if (left_size == 0 || right_size == 0)
        return parent_entropy;

    size_t size = left_size + right_size;

    double e_left = entropy(left);
    double e_right = entropy(right);

    double ratio_left = (double)left_size / size;
    double ratio_right = (double)right_size / size;

    return parent_entropy - (ratio_left * e_left + ratio_right * e_right);
}
