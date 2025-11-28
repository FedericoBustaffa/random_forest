#include "dataframe.hpp"

#include <regex>
#include <set>
#include <unordered_map>

DataFrame::DataFrame(const std::vector<std::string>& content, size_t rows,
                     size_t cols)
    : m_Content(content), m_Rows(rows), m_Cols(cols)
{
}

const std::string& DataFrame::operator()(size_t row, size_t col) const
{
    assert(row >= 0 && row < m_Rows);
    assert(col >= 0 && col < m_Cols);
    return m_Content[row * m_Cols + col];
}

std::pair<std::vector<std::vector<double>>, std::vector<uint32_t>> DataFrame::
    to_vector() const
{
    std::vector<std::vector<double>> data(m_Cols);

    // regex to capture every possible numerical type
    std::regex numerical(
        R"(^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$)");
    for (size_t i = 0; i < m_Cols; i++)
    {
        if (std::regex_search(m_Content[i], numerical))
        {
            for (size_t j = 0; j < m_Rows; j++)
                data[i].push_back(std::stod(m_Content[j * m_Cols + i]));
        }
        else
        {
            std::set<std::string> possible_values;
            for (size_t j = 0; j < m_Rows; j++)
                possible_values.emplace(m_Content[j * m_Cols + i]);

            std::unordered_map<std::string, double> encoder;
            double num = 0.0;
            for (const auto& pv : possible_values)
            {
                encoder[pv] = (double)num;
                num += 1.0;
            }

            for (size_t j = 0; j < m_Rows; j++)
                data[i].push_back(encoder[m_Content[j * m_Cols + i]]);
        }
    }

    std::vector<std::vector<double>> X(data.begin(), data.end() - 1);
    std::vector<uint32_t> y(data.back().begin(), data.back().end());

    return {X, y};
}

DataFrame::~DataFrame() {}
