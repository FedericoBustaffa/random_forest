#include "tree_functions.hpp"

#include <cmath>

#include "counter.hpp"
#include "utils.hpp"

double entropy(const Counter& counters)
{
    size_t size = 0;
    for (size_t i = 0; i < counters.size(); ++i)
        size += counters[i];

    double e = 0.0;
    double proportion;
    for (size_t i = 0; i < counters.size(); ++i)
    {
        if (counters[i] == 0)
            continue;

        proportion = (double)counters[i] / size;
        e -= proportion * std::log2(proportion);
    }

    return e;
}

double entropy(const std::vector<uint32_t>& y,
               const std::vector<size_t>& indices)
{
    return entropy(count(y, indices));
}

double informationGain(double parent_entropy, const Counter& left,
                       const Counter& right)
{
    size_t left_size = 0;
    for (size_t i = 0; i < left.size(); ++i)
        left_size += left[i];

    size_t right_size = 0;
    for (size_t i = 0; i < right.size(); ++i)
        right_size += right[i];

    if (left_size == 0 || right_size == 0)
        return parent_entropy;

    size_t size = left_size + right_size;

    double e_left = entropy(left);
    double e_right = entropy(right);

    double ratio_left = (double)left_size / size;
    double ratio_right = (double)right_size / size;

    return parent_entropy - (ratio_left * e_left + ratio_right * e_right);
}
