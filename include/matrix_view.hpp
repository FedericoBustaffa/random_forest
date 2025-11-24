#ifndef MATRIX_VIEW_HPP
#define MATRIX_VIEW_HPP

#include <cstddef>

#include "vector_view.hpp"

class MatrixView
{
public:
    MatrixView(const double* data, size_t rows, size_t cols,
               const std::vector<size_t>& indices)
        : m_View(data), m_Rows(rows), m_Cols(cols), m_Indices(indices)
    {
    }

    MatrixView(const double* data, size_t rows, size_t cols)
        : m_View(data), m_Rows(rows), m_Cols(cols), m_Indices(m_Rows)
    {
        for (size_t i = 0; i < m_Rows; i++)
            m_Indices[i] = i;
    }

    MatrixView(const MatrixView& other)
        : m_View(other.m_View), m_Rows(other.m_Rows), m_Cols(other.m_Cols),
          m_Indices(other.m_Indices)
    {
    }

    MatrixView(MatrixView&& other) noexcept
        : m_View(other.m_View), m_Rows(other.m_Rows), m_Cols(other.m_Cols),
          m_Indices(std::move(other.m_Indices))
    {
        other.m_View = nullptr;
        other.m_Rows = 0;
        other.m_Cols = 0;
    }

    MatrixView& operator=(const MatrixView& other)
    {
        if (this != &other)
        {
            m_View = other.m_View;
            m_Rows = other.m_Rows;
            m_Cols = other.m_Cols;
            m_Indices = other.m_Indices;
        }

        return *this;
    }

    MatrixView& operator=(MatrixView&& other) noexcept
    {
        if (this != &other)
        {
            m_View = other.m_View;
            m_Rows = other.m_Rows;
            m_Cols = other.m_Cols;
            m_Indices = std::move(other.m_Indices);

            other.m_View = nullptr;
            other.m_Rows = 0;
            other.m_Cols = 0;
        }

        return *this;
    }

    virtual inline size_t rows() const { return m_Rows; }

    virtual inline size_t cols() const { return m_Cols; }

    virtual inline size_t size() const { return m_Rows * m_Cols; }

    virtual VectorView operator()(size_t col) const
    {
        return VectorView(m_View + col, m_Rows, m_Cols, m_Indices);
    }

    virtual inline VectorView operator[](size_t row) const
    {
        return VectorView(m_View + m_Indices[row] * m_Cols, m_Cols);
    }

    MatrixView operator[](const std::vector<size_t>& indices) const
    {
        std::vector<size_t> new_indices(indices.size());
        for (size_t i = 0; i < indices.size(); i++)
            new_indices[i] = m_Indices[indices[i]];

        return MatrixView(m_View, new_indices.size(), m_Cols, new_indices);
    }

    MatrixView operator[](const Mask& mask) const
    {
        std::vector<size_t> indices;
        for (size_t i = 0; i < m_Rows; i++)
            if (mask[i])
                indices.push_back(m_Indices[i]);

        return MatrixView(m_View, indices.size(), m_Cols, indices);
    }

protected:
    const double* m_View = nullptr;
    size_t m_Rows = 0, m_Cols = 0;
    std::vector<size_t> m_Indices;
};

#endif
