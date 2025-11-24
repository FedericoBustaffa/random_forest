#include <cstdio>

#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc != 2)
    {
        std::printf("USAGE: %s <filepath>\n", argv[0]);
        return 1;
    }

    auto [X, y] = read_csv(argv[1]);
    for (size_t i = 0; i < X.rows(); i++)
    {
        for (size_t j = 0; j < X.cols(); j++)
            std::printf("%.2f ", (double)X[i][j]);

        std::printf("%.2f\n", (double)y[i]);
    }

    return 0;
}
