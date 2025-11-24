#include "matrix_view.hpp"

#include "vector_view.hpp"

MatrixView::MatrixView(const double* data, size_t rows, size_t cols,
                       const std::vector<size_t>& indices)
    : m_View(data), m_Rows(rows), m_Cols(cols), m_Indices(indices)
{
}

MatrixView::MatrixView(const double* data, size_t rows, size_t cols)
    : m_View(data), m_Rows(rows), m_Cols(cols), m_Indices(m_Rows)
{
    for (size_t i = 0; i < m_Rows; i++)
        m_Indices[i] = i;
}

MatrixView::MatrixView(const MatrixView& other)
    : m_View(other.m_View), m_Rows(other.m_Rows), m_Cols(other.m_Cols),
      m_Indices(other.m_Indices)
{
}

MatrixView::MatrixView(MatrixView&& other) noexcept
    : m_View(other.m_View), m_Rows(other.m_Rows), m_Cols(other.m_Cols),
      m_Indices(std::move(other.m_Indices))
{
    other.m_View = nullptr;
    other.m_Rows = 0;
    other.m_Cols = 0;
}

MatrixView& MatrixView::operator=(const MatrixView& other)
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

MatrixView& MatrixView::operator=(MatrixView&& other) noexcept
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

VectorView MatrixView::operator()(size_t col) const
{
    return VectorView(m_View + col, m_Rows, m_Cols, m_Indices);
}

MatrixView MatrixView::operator[](const std::vector<size_t>& indices) const
{
    std::vector<size_t> new_indices(indices.size());
    for (size_t i = 0; i < indices.size(); i++)
        new_indices[i] = m_Indices[indices[i]];

    return MatrixView(m_View, new_indices.size(), m_Cols, new_indices);
}

MatrixView MatrixView::operator[](const Mask& mask) const
{
    std::vector<size_t> indices;
    for (size_t i = 0; i < m_Rows; i++)
        if (mask[i])
            indices.push_back(m_Indices[i]);

    return MatrixView(m_View, indices.size(), m_Cols, indices);
}
