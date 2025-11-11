#include "column.hpp"

#include <algorithm>
#include <regex>

#include "utils.hpp"

column::column(const std::string& header,
               const std::vector<std::string>& content)
    : m_header(header), m_content(content)
{
    // auto type deduction
    std::regex numerical(
        R"(^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$)");

    if (std::regex_search(content[0], numerical))
        m_type = datatype::numerical;
    else
    {
        m_type = datatype::categorical;

        // build a mapping between categorical string and intger
        std::vector<std::string> uniques = content;
        std::sort(uniques.begin(), uniques.end());
        uniques.erase(std::unique(uniques.begin(), uniques.end()),
                      uniques.end());

        for (size_t i = 0; i < uniques.size(); i++)
            m_dict[uniques[i]] = (double)i;
    }
}

double column::get(size_t i)
{
    if (m_type == datatype::categorical)
        return m_dict[m_content[i]];
    else
        return std::stod(m_content[i]);
}

std::vector<double> column::to_vec() const
{
    if (m_type == datatype::categorical)
        return encode(m_content);

    return convert(m_content);
}

column::~column() {}
