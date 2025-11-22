#include <cstdio>
#include <random>

#include "tensor.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0, 1);

    std::vector<double> data;
    for (size_t i = 0; i < 4; i++)
        data.push_back(dist(rng));

    double x = 10.0;
    Tensor s(&x, {});
    std::printf("scalar: %.2f\n", (double)s);

    Tensor v(data.data(), {4});
    std::printf("vector: [");
    for (size_t i = 0; i < v.size(); i++)
        std::printf("%.2f ", (double)v[i]);
    std::printf("]\n");

    Tensor m(data.data(), {2, 2});
    std::printf("matrix\n");
    for (size_t i = 0; i < m.shape()[0]; i++)
    {
        for (size_t j = 0; j < m.shape()[1]; j++)
            std::printf("%.2f ", (double)(m[i][j]));
        std::printf("\n");
    }

    std::printf("argsorted\n");
    std::vector<size_t> indices = argsort(v);
    TensorView sorted = v[indices];
    for (size_t i = 0; i < sorted.size(); i++)
        std::printf("%.2f\n", (double)sorted[i]);

    TensorView col = m(0, 1);
    std::printf("sliced\n");
    for (size_t i = 0; i < col.size(); i++)
        std::printf("%.2f\n", (double)col[i]);

    std::vector<bool> mask;
    for (size_t i = 0; i < v.size(); i++)
        mask.push_back(v[i] < 0.5);

    TensorView masked = v[mask];
    std::printf("masked\n");
    for (size_t i = 0; i < masked.size(); i++)
        std::printf("%.2f\n", (double)masked[i]);

    return 0;
}
