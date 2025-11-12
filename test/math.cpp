#include <cstdio>
#include <random>

#include "tensor.hpp"

int main(int argc, const char** argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0, 1);

    std::vector<double> data;
    for (size_t i = 0; i < 16; i++)
        data.push_back(dist(rng));

    Tensor m(data.data(), {4, 4});
    std::printf("matrix\n");
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
            std::printf("%.2f ", (double)m[i][j]);
        std::printf("\n");
    }

    Tensor v = m[1];
    std::printf("vector\n");
    for (size_t i = 0; i < 4; i++)
        std::printf("%.2f\n", (double)v[i]);

    Tensor s = v[3];
    std::printf("scalar: %.2f\n", (double)s);

    return 0;
}
