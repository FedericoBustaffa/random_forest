#include "dataframe.hpp"

#include <algorithm>
#include <regex>

dataframe::dataframe(const std::vector<std::string>& content,
                     const std::vector<std::string>& headers)
    : m_rows(content.size() / headers.size()), m_cols(headers.size()),
      m_headers(headers), m_content(content)
{
    // automatic type deduction
    std::regex numerical(
        R"(^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$)");

    for (size_t i = 0; i < headers.size(); i++)
    {
        if (std::regex_search(m_content[i], numerical))
            m_fields.emplace_back(field::numerical);
        else
            m_fields.emplace_back(field::categorical);
    }
}

std::vector<std::string_view> dataframe::operator[](
    const std::string& header) const
{
    auto h = std::find(m_headers.begin(), m_headers.end(), header);
    if (h == m_headers.end())
        return {};

    size_t idx = std::distance(m_headers.begin(), h);
    std::vector<std::string_view> column;
    column.reserve(m_cols);

    for (size_t i = 0; i < m_rows; i++)
        column.push_back(m_content[i * m_cols + idx]);

    return column;
}

std::vector<double> dataframe::to_vector() const
{
    std::vector<double> v;
    v.reserve(m_rows * m_cols);

    for (size_t i = 0; i < m_rows; i++)
    {
        for (size_t j = 0; j < m_cols; j++)
        {
            if (m_fields[j] == field::numerical)
                v.push_back(
                    std::strtod(m_content[i * m_cols + j].c_str(), nullptr));
        }
    }
}

dataframe::~dataframe() {}
