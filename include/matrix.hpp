#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "matrix_view.hpp"

class Matrix : public MatrixView
{
public:
    Matrix(const double* data, size_t rows, size_t cols);

    Matrix(const Matrix& other);

    Matrix(Matrix&& other) noexcept;

    Matrix& operator=(const Matrix& other);

    Matrix& operator=(Matrix&& other) noexcept;

    ~Matrix();

private:
    double* m_Data;
};

#endif
