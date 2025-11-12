#include "matrix.hpp"

#include <cstring>

Matrix::Matrix(const double* data, size_t rows, size_t cols)
    : m_Rows(rows), m_Cols(cols)
{
    m_Data = new double[rows * cols];
    std::memcpy(m_Data, data, rows * cols * sizeof(double));
}

Matrix::Matrix(const Matrix& other) : m_Rows(other.m_Rows), m_Cols(other.m_Cols)
{
    m_Data = new double[m_Rows * m_Cols];
    std::memcpy(m_Data, other.m_Data, m_Rows * m_Cols * sizeof(double));
}

Matrix::Matrix(Matrix&& other)
    : m_Rows(other.m_Rows), m_Cols(other.m_Cols), m_Data(other.m_Data)
{
    other.m_Rows = 0;
    other.m_Cols = 0;
    other.m_Data = nullptr;
}

Matrix::~Matrix() { delete[] m_Data; }
