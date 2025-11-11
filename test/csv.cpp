#include <iostream>

#include "csv.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc != 2)
    {
        std::printf("USAGE: %s <filepath>\n", argv[0]);
        return 1;
    }

    dataframe df = read_csv(argv[1], {"sepal_length", "sepal_width",
                                      "petal_length", "petal_width", "class"});

    std::printf("--- info ---\n");
    std::printf("shape: (%lu, %lu)", df.nrows(), df.ncolumns());

    for (const auto& h : df.headers())
        std::cout << h << ": " << df[h].type() << std::endl;

    auto [X, y] = build_dataset(df, "class");
    for (size_t i = 0; i < X.size(); i++)
    {
        for (size_t j = 0; j < X[i].size(); j++)
            std::printf("%.2f ", X[i][j]);
        std::cout << std::endl;
    }

    return 0;
}
