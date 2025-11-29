#ifndef DATAFRAME_HPP
#define DATAFRAME_HPP

#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class DataFrame
{
public:
    DataFrame(const std::vector<std::string>& content, size_t rows,
              size_t cols);

    inline size_t rows() const { return m_Rows; }

    inline size_t cols() const { return m_Cols; }

    const std::string& operator()(size_t row, size_t col) const;

    std::pair<std::vector<std::vector<double>>, std::vector<uint32_t>>
    to_vector();

    ~DataFrame();

private:
    enum class DataType
    {
        Numerical,
        Categorical
    };

private:
    std::vector<std::string> m_Content;
    size_t m_Rows, m_Cols;

    std::vector<DataType> m_DataTypes;
    std::vector<std::unordered_map<std::string, double>> m_Encoders;
};

#endif
