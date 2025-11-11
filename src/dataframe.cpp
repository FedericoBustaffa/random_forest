#include "dataframe.hpp"

#include <algorithm>
#include <stdexcept>

dataframe::dataframe(const std::vector<std::vector<std::string>>& content,
                     const std::vector<std::string>& headers)
    : m_headers(headers)
{
    for (size_t i = 0; i < content.size(); i++)
        m_fields.emplace_back(headers[i], content[i]);
}

column dataframe::operator[](const std::string_view& header) const
{
    auto h = std::find(m_headers.begin(), m_headers.end(), header);
    if (h == m_headers.end())
        throw std::runtime_error("no such header");

    size_t idx = std::distance(m_headers.begin(), h);

    return m_fields[idx];
}

std::vector<double> dataframe::to_vec() const
{
    std::vector<double> data;
    size_t rows = m_fields[0].size();
    size_t cols = m_fields.size();

    data.reserve(rows * cols);

    for (const auto& f : m_fields)
    {
        for (const auto& v : f.to_vec())
            data.push_back(v);
    }

    return data;
}

dataframe::~dataframe() {}
