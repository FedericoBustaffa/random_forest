#include "tree_functions.hpp"

#include <cmath>

#include "counter.hpp"
#include "utils.hpp"

// Compute entropy from class counters
float entropy(const Counter& counters)
{
    size_t size = 0;
    for (size_t i = 0; i < counters.size(); ++i)
        size += counters[i]; // total number of samples

    float e = 0.0;
    float proportion;
    for (size_t i = 0; i < counters.size(); ++i)
    {
        if (counters[i] == 0)
            continue; // skip empty classes

        proportion = (float)counters[i] / size;
        e -= proportion * std::log2(proportion);
    }

    return e;
}

// Compute entropy on a subset of y
float entropy(const std::vector<uint8_t>& y, const std::vector<size_t>& indices)
{
    return entropy(count(y, indices));
}

// Compute information gain from parent and child counters
float informationGain(float parent_entropy, const Counter& left,
                      const Counter& right)
{
    size_t left_size = 0;
    for (size_t i = 0; i < left.size(); ++i)
        left_size += left[i]; // number of samples in left child

    size_t right_size = 0;
    for (size_t i = 0; i < right.size(); ++i)
        right_size += right[i]; // number of samples in right child

    if (left_size == 0 || right_size == 0)
        return parent_entropy; // no split

    size_t size = left_size + right_size;

    float e_left = entropy(left);
    float e_right = entropy(right);

    float ratio_left = (float)left_size / size;
    float ratio_right = (float)right_size / size;

    return parent_entropy - (ratio_left * e_left + ratio_right * e_right);
}
