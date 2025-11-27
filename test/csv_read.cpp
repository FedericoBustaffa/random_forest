#include <iostream>

#include "csv.hpp"
#include "dataframe.hpp"

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

    auto [X, y] = df.toVector();
    for (size_t i = 0; i < X[0].size(); i++)
    {
        for (size_t j = 0; j < X.size(); j++)
            std::printf("%.2f ", X[j][i]);
        std::printf("%u\n", y[i]);
    }

    return 0;
}
