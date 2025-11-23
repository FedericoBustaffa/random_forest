#include <cstdio>
#include <random>

#include "tensor.hpp"
#include "utils.hpp"

Tensor init(size_t m, size_t n)
{

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0, 1);

    std::vector<double> data;
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            data.push_back(dist(rng));

    return Tensor(data.data(), {m, n});
}

void square_bracket(const Tensor& t)
{
    std::printf("matrix:\n");
    for (size_t i = 0; i < t.shape()[0]; i++)
    {
        std::printf("[ ");
        for (size_t j = 0; j < t.shape()[1]; j++)
            std::printf("%.2f ", (double)(t[i][j]));
        std::printf("]\n");
    }

    TensorView v = t[1];
    std::printf("vector: [ ");
    for (size_t i = 0; i < v.size(); i++)
        std::printf("%.2f ", (double)v[i]);
    std::printf("]\n");

    TensorView s = v[1];
    std::printf("scalar: %.2f\n", (double)s);
}

void slice(const Tensor& t)
{
    TensorView col = t(1, 1);
    std::printf("sliced: [ ");
    for (size_t i = 0; i < col.size(); i++)
        std::printf("%.2f ", (double)col[i]);
    std::printf("]\n");

    TensorView s = col[1];
    std::printf("scalar: %.2f\n", (double)s);
}

void sorting(const Tensor& t)
{
    TensorView v = t[1];
    std::printf("argsorted: [ ");
    std::vector<size_t> indices = argsort(v);
    for (size_t i = 0; i < v.size(); i++)
        std::printf("%.2f ", (double)v[indices[i]]);
    std::printf("]\n");
}

int main(int argc, const char** argv)
{
    Tensor t = init(3, 2);

    square_bracket(t);
    slice(t);
    sorting(t);

    return 0;
}
