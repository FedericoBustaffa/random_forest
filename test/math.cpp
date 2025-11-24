#include <cstdio>
#include <random>

#include "matrix.hpp"
#include "utils.hpp"
#include "vector_view.hpp"

Matrix init(size_t m, size_t n)
{

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0, 1);

    std::vector<double> data;
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            data.push_back(dist(rng));

    return Matrix(data.data(), m, n);
}

void square_bracket(const Matrix& t)
{
    std::printf("matrix:\n");
    for (size_t i = 0; i < t.rows(); i++)
    {
        std::printf("[ ");
        for (size_t j = 0; j < t.cols(); j++)
            std::printf("%.2f ", t[i][j]);
        std::printf("]\n");
    }

    VectorView v = t[1];
    std::printf("vector: [ ");
    for (size_t i = 0; i < v.size(); i++)
        std::printf("%.2f ", v[i]);
    std::printf("]\n");

    double s = v[1];
    std::printf("scalar: %.2f\n", s);
}

void slice(const Matrix& m)
{
    VectorView col = m(1);
    std::printf("sliced: [ ");
    for (size_t i = 0; i < col.size(); i++)
        std::printf("%.2f ", col[i]);
    std::printf("]\n");

    double s = col[1];
    std::printf("scalar: %.2f\n", s);
}

void sorting(const Matrix& t)
{
    VectorView v = t[1];
    std::printf("argsorted: [ ");
    std::vector<size_t> indices = argsort(v);
    v = v[indices];
    for (size_t i = 0; i < v.size(); i++)
        std::printf("%.2f ", (double)v[i]);
    std::printf("]\n");

    VectorView u = t(1);
    std::printf("argsorted: [ ");
    indices = argsort(u);
    u = u[indices];
    for (size_t i = 0; i < u.size(); i++)
        std::printf("%.2f ", (double)u[i]);
    std::printf("]\n");
}

void masking(const Matrix& m)
{
    std::vector<size_t> indices = argsort(m(0));
    MatrixView m2 = m[indices];
    std::printf("matrix:\n");
    for (size_t i = 0; i < m2.rows(); i++)
    {
        std::printf("[ ");
        for (size_t j = 0; j < m2.cols(); j++)
            std::printf("%.2f ", m2[i][j]);
        std::printf("]\n");
    }

    m2 = m[m(0) <= 0.8];
    std::printf("matrix:\n");
    for (size_t i = 0; i < m2.rows(); i++)
    {
        std::printf("[ ");
        for (size_t j = 0; j < m2.cols(); j++)
            std::printf("%.2f ", m2[i][j]);
        std::printf("]\n");
    }

    indices = argsort(m2(1));
    MatrixView m3 = m2[indices];
    m3 = m2[m2(1) <= 0.5];
    std::printf("matrix:\n");
    for (size_t i = 0; i < m3.rows(); i++)
    {
        std::printf("[ ");
        for (size_t j = 0; j < m3.cols(); j++)
            std::printf("%.2f ", m3[i][j]);
        std::printf("]\n");
    }
}

int main(int argc, const char** argv)
{
    Matrix t = init(4, 5);

    square_bracket(t);
    slice(t);
    sorting(t);
    masking(t);

    return 0;
}
