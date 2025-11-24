#include "matrix.hpp"

#include <cstring>
#include <utility>

Matrix::Matrix(const double* data, size_t rows, size_t cols)
    : MatrixView(nullptr, rows, cols)
{
    m_Data = new double[rows * cols];
    std::memcpy(m_Data, data, rows * cols * sizeof(double));
    m_View = m_Data;
}

Matrix::Matrix(const Matrix& other)
    : MatrixView(nullptr, other.m_Rows, other.m_Cols)
{
    m_Data = new double[m_Rows * m_Cols];
    std::memcpy(m_Data, other.m_Data, m_Rows * m_Cols * sizeof(double));
    m_View = m_Data;
}

Matrix::Matrix(Matrix&& other) noexcept
    : MatrixView(std::move(other)), m_Data(other.m_Data)
{
    m_View = m_Data;
    other.m_Data = nullptr;
}

Matrix& Matrix::operator=(const Matrix& other)
{
    if (this != &other)
    {
        delete[] m_Data;

        MatrixView::operator=(other);
        m_Data = new double[m_Rows * m_Cols];
        std::memcpy(m_Data, other.m_Data, m_Rows * m_Cols * sizeof(double));

        m_View = m_Data;
    }

    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept
{
    if (this != &other)
    {
        MatrixView::operator=(std::move(other));

        delete[] m_Data;
        m_Data = other.m_Data;
        m_View = m_Data;

        other.m_Data = nullptr;
    }

    return *this;
}

Matrix::~Matrix() { delete[] m_Data; }
