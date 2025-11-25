#ifndef DATAFRAME_HPP
#define DATAFRAME_HPP

#include <cassert>
#include <string>
#include <vector>

class DataFrame
{
public:
    DataFrame(const std::vector<std::string>& content, size_t rows,
              size_t cols);

    inline size_t rows() const { return m_Rows; }

    inline size_t cols() const { return m_Cols; }

    const std::string& operator()(size_t row, size_t col) const;

    std::vector<std::vector<double>> toVector() const;

    ~DataFrame();

private:
    std::vector<std::string> m_Content;
    size_t m_Rows, m_Cols;
};

#endif
