#include "dataframe.hpp"

#include <regex>
#include <set>

DataFrame::DataFrame(const std::vector<std::string>& content, size_t rows,
                     size_t cols)
    : m_Content(content), m_Rows(rows), m_Cols(cols)
{
    // regex to capture every possible numerical type
    std::regex numerical(
        R"(^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$)");

    for (size_t i = 0; i < m_Cols; i++)
    {
        if (std::regex_search(m_Content[i], numerical))
        {
            m_DataTypes.push_back(DataType::Numerical);
            m_Encoders.push_back({});
        }
        else
        {
            std::set<std::string> possible_values;
            for (size_t j = 0; j < m_Rows; j++)
                possible_values.emplace(m_Content[j * m_Cols + i]);

            std::unordered_map<std::string, double> encoder;
            double counter = 0.0;
            for (const auto& pv : possible_values)
            {
                encoder[pv] = (double)counter;
                counter += 1.0;
            }

            m_DataTypes.push_back(DataType::Categorical);
            m_Encoders.push_back(encoder);
        }
    }
}

const std::string& DataFrame::operator()(size_t row, size_t col) const
{
    assert(row >= 0 && row < m_Rows);
    assert(col >= 0 && col < m_Cols);
    return m_Content[row * m_Cols + col];
}

std::pair<std::vector<std::vector<double>>, std::vector<uint32_t>> DataFrame::
    to_vector()
{
    std::vector<std::vector<double>> X(m_Rows);
    std::vector<uint32_t> y(m_Rows);

    for (size_t i = 0; i < m_Rows; i++)
    {
        for (size_t j = 0; j < m_Cols - 1; j++)
        {
            if (m_DataTypes[j] == DataType::Numerical)
                X[i].push_back(std::stod(m_Content[i * m_Cols + j]));
            else
                X[i].push_back(m_Encoders[j][m_Content[i * m_Cols + j]]);
        }

        if (m_DataTypes[m_Cols - 1] == DataType::Numerical)
            y[i] = std::stoul(m_Content[i * m_Cols + (m_Cols - 1)]);
        else
        {
            const std::string& last = m_Content[i * m_Cols + (m_Cols - 1)];
            y[i] = m_Encoders[m_Cols - 1][last];
        }
    }

    return {X, y};
}

DataFrame::~DataFrame() {}
