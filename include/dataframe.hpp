#ifndef DATAFRAME_HPP
#define DATAFRAME_HPP

#include <string>
#include <vector>

class dataframe
{
public:
    enum class field
    {
        numerical,
        categorical
    };

public:
    dataframe(const std::vector<std::string>& content,
              const std::vector<std::string>& headers);

    inline size_t rows() const { return m_rows; }

    inline size_t columns() const { return m_cols; }

    inline bool empty() const { return m_content.empty(); }

    inline const std::vector<std::string>& headers() const { return m_headers; }

    inline const std::vector<field>& fields() const { return m_fields; }

    std::vector<std::string_view> operator[](const std::string& header) const;

    std::vector<double> to_vector() const;

    ~dataframe();

private:
    size_t m_rows, m_cols;
    std::vector<std::string> m_headers;
    std::vector<field> m_fields;
    std::vector<std::string> m_content;
};

#endif
