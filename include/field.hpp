#ifndef FIELD_HPP
#define FIELD_HPP

#include <ostream>
#include <string>
#include <vector>

class field
{
public:
    enum class datatype
    {
        numerical,
        categorical
    };

public:
    field(const std::string& header, const std::vector<std::string>& content);

    inline datatype type() const { return m_type; }

    inline const std::string& header() const { return m_header; }

    inline size_t size() const { return m_content.size(); }

    inline const std::string& operator[](size_t i) const
    {
        return m_content[i];
    }

    std::vector<double> as_double() const;

    ~field();

private:
    datatype m_type;
    std::string m_header;
    std::vector<std::string> m_content;
};

inline std::ostream& operator<<(std::ostream& os, field::datatype dt)
{
    if (dt == field::datatype::numerical)
        os << "numerical";
    else
        os << "categorical";

    return os;
}

#endif
