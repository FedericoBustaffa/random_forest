#ifndef MATRIX_VIEW_HPP
#define MATRIX_VIEW_HPP

#include <cstddef>

#include "vector_view.hpp"

class MatrixView
{
public:
    MatrixView(const double* data, size_t rows, size_t cols,
               const std::vector<size_t>& indices);

    MatrixView(const double* data, size_t rows, size_t cols);

    MatrixView(const MatrixView& other);

    MatrixView(MatrixView&& other) noexcept;

    MatrixView& operator=(const MatrixView& other);

    MatrixView& operator=(MatrixView&& other) noexcept;

    virtual inline size_t rows() const { return m_Rows; }

    virtual inline size_t cols() const { return m_Cols; }

    virtual inline size_t size() const { return m_Rows * m_Cols; }

    virtual VectorView operator()(size_t col) const;

    virtual inline VectorView operator[](size_t row) const
    {
        return VectorView(m_View + m_Indices[row] * m_Cols, m_Cols);
    }

    MatrixView operator[](const std::vector<size_t>& indices) const;

    MatrixView operator[](const Mask& mask) const;

protected:
    const double* m_View = nullptr;
    size_t m_Rows = 0, m_Cols = 0;
    std::vector<size_t> m_Indices;
};

#endif
