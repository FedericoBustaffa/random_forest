#include "field.hpp"

#include <regex>

#include "utils.hpp"

field::field(const std::string& header, const std::vector<std::string>& content)
    : m_header(header), m_content(content)
{
    // auto type deduction
    std::regex numerical(
        R"(^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$)");

    if (std::regex_search(content[0], numerical))
        m_type = datatype::numerical;
    else
        m_type = datatype::categorical;
}

std::vector<double> field::as_double() const
{
    if (m_type == datatype::categorical)
        return encode(m_content);

    return convert(m_content);
}

field::~field() {}
