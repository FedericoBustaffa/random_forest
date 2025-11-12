#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>

class Matrix
{
public:
    Matrix(const double* data, size_t rows, size_t cols);

    Matrix(const Matrix& other);

    Matrix(Matrix&& other);

    ~Matrix();

private:
    size_t m_Rows, m_Cols;
    double* m_Data;
};

#endif
