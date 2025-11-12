#ifndef DATAFRAME_HPP
#define DATAFRAME_HPP

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "matrix.hpp"
#include "vector.hpp"

enum class DataType
{
    numerical,
    categorical
};

class DataFrame
{
public:
    DataFrame(const std::vector<std::string>& content,
              const std::vector<std::string>& headers);

    inline size_t rows() const { return m_Rows; }

    inline size_t columns() const { return m_Cols; }

    inline const std::vector<std::string>& headers() const { return m_Headers; }

    inline const std::vector<DataType>& datatypes() const
    {
        return m_DataTypes;
    }

    inline const std::vector<std::string>& content() const { return m_Content; }

    std::pair<Matrix, Vector> toData() const;

    ~DataFrame();

private:
    size_t m_Rows, m_Cols;
    std::vector<std::string> m_Headers;
    std::vector<DataType> m_DataTypes;
    std::vector<std::string> m_Content;
};

inline std::ostream& operator<<(std::ostream& os, DataType dt)
{
    if (dt == DataType::numerical)
        os << "numerical";
    else
        os << "categorical";

    return os;
}

#endif
