#include "utils.hpp"

#include <algorithm>
#include <numeric>

#include "view.hpp"

std::vector<size_t> argsort(const View<double>& v)
{
    std::vector<size_t> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);

    auto compare = [&v](const auto& a, const auto& b) { return v[a] < v[b]; };
    std::sort(indices.begin(), indices.end(), compare);

    return indices;
}

double accuracy(const std::vector<double>& predictions,
                const std::vector<double>& correct)
{
    double counter = 0.0;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == correct[i])
            counter++;
    }

    return counter / predictions.size();
}
