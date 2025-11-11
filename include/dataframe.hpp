#ifndef DATAFRAME_HPP
#define DATAFRAME_HPP

#include <string>
#include <vector>

#include "column.hpp"

class dataview
{
public:
    dataview();
    ~dataview();

private:
};

class dataframe
{
public:
    dataframe(const std::vector<std::vector<std::string>>& content,
              const std::vector<std::string>& headers);

    inline size_t nrows() const { return m_fields[0].size(); }

    inline size_t ncolumns() const { return m_fields.size(); }

    inline bool empty() const { return m_fields.empty(); }

    inline const std::vector<std::string>& headers() const { return m_headers; }

    inline std::vector<column> fields() const { return m_fields; }

    inline column operator[](size_t i) const { return m_fields[i]; }

    column operator[](const std::string_view& header) const;

    std::vector<double> to_vec() const;

    ~dataframe();

private:
    std::vector<std::string> m_headers;
    std::vector<column> m_fields;
};

#endif
