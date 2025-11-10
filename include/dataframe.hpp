#ifndef DATAFRAME_HPP
#define DATAFRAME_HPP

#include <string>
#include <vector>

#include "field.hpp"

class dataframe
{
public:
    dataframe(const std::vector<std::vector<std::string>>& content,
              const std::vector<std::string>& headers);

    inline size_t rows() const { return m_fields[0].size(); }

    inline size_t columns() const { return m_fields.size(); }

    inline bool empty() const { return m_fields.empty(); }

    inline std::vector<std::string_view> headers() const { return m_headers; }

    inline std::vector<field> fields() const { return m_fields; }

    field operator[](const std::string_view& header) const;

    ~dataframe();

private:
    std::vector<std::string_view> m_headers;
    std::vector<field> m_fields;
};

#endif
