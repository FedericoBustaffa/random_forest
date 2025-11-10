#include "utils.hpp"

#include <algorithm>

std::vector<double> encode(const std::vector<std::string>& labels)
{
    std::vector<std::string_view> unique_values;
    std::vector<double> encoded;
    encoded.reserve(labels.size());

    for (const auto& l : labels)
    {
        auto it = std::find(unique_values.begin(), unique_values.end(), l);
        if (it == unique_values.end())
        {
            unique_values.push_back(l);
            encoded.push_back(unique_values.size() - 1);
        }
        else
            encoded.push_back(std::distance(unique_values.begin(), it));
    }

    return encoded;
}

std::vector<double> convert(const std::vector<std::string>& values)
{
    std::vector<double> converted;

    for (const auto& v : values)
        converted.push_back(std::stod(v));

    return converted;
}
