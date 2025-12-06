#include <iostream>

#include "dataframe.hpp"
#include "io.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc != 2)
    {
        std::printf("USAGE: %s <filepath>\n", argv[0]);
        return 1;
    }

    DataFrame df = read_csv(argv[1]);
    for (size_t i = 0; i < df.rows(); i++)
    {
        for (size_t j = 0; j < df.cols(); j++)
            std::cout << df(i, j) << " ";
        std::cout << std::endl;
    }

    auto [X, y] = df.to_vector();
    for (size_t i = 0; i < X.size(); i++)
    {
        for (size_t j = 0; j < X[i].size(); j++)
            std::printf("%.2f ", X[i][j]);
        std::printf("%u\n", y[i]);
    }

    auto T = transpose(X);
    for (size_t i = 0; i < T[0].size(); i++)
    {
        for (size_t j = 0; j < T.size(); j++)
            std::printf("%.2f ", T[j][i]);
        std::printf("\n");
    }

    return 0;
}
