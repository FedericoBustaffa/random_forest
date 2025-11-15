#include "utils.hpp"

#include <algorithm>
#include <numeric>

std::vector<size_t> argsort(const TensorView& t)
{
    std::vector<size_t> indices(t.size());
    std::iota(indices.begin(), indices.end(), 0);

    auto compare = [&t](const auto& a, const auto& b) { return t[a] < t[b]; };
    std::sort(indices.begin(), indices.end(), compare);

    return indices;
}

double accuracy(const TensorView& predictions, const TensorView& correct)
{
    double counter = 0.0;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == correct[i])
            counter++;
    }

    return counter / predictions.size();
}
