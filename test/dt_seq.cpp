#include <iostream>

#include "dataframe.hpp"
#include "decision_tree.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc != 2)
    {
        std::printf("USAGE: %s <filepath>\n", argv[0]);
        return 1;
    }

    DataFrame df = read_csv(argv[1]);
    std::vector<std::vector<double>> data = df.toVector();
    std::vector<std::vector<double>> X(data.begin(), data.end() - 1);
    std::vector<double> y = data.back();

    DecisionTree tree;
    tree.fit(X, y);

    return 0;
}
