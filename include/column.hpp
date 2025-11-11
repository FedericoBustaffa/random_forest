#ifndef COLUMN_HPP
#define COLUMN_HPP

#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

class column
{
public:
    enum class datatype
    {
        numerical,
        categorical
    };

public:
    column(const std::string& header, const std::vector<std::string>& content);

    inline datatype type() const { return m_type; }

    inline const std::string& header() const { return m_header; }

    inline size_t size() const { return m_content.size(); }

    inline const std::string& operator[](size_t i) const
    {
        return m_content[i];
    }

    double get(size_t i);

    std::vector<double> to_vec() const;

    ~column();

private:
    datatype m_type;
    std::string m_header;
    std::vector<std::string> m_content;
    std::unordered_map<std::string, double> m_dict;
};

inline std::ostream& operator<<(std::ostream& os, column::datatype dt)
{
    if (dt == column::datatype::numerical)
        os << "numerical";
    else
        os << "categorical";

    return os;
}

#endif
