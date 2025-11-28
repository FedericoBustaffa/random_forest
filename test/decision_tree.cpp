#include <cstdio>

#include "csv.hpp"
#include "dataframe.hpp"
#include "decision_tree.hpp"
#include "timer.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc != 3)
    {
        std::printf("USAGE: %s <max_depth> <filepath>\n", argv[0]);
        return 1;
    }

    size_t max_depth = std::stoull(argv[1]);

    DataFrame df = read_csv(argv[2]);
    auto [X, y] = df.toVector();

    DecisionTree tree(max_depth);
    Timer<milli> timer;
    timer.start();
    tree.fit(X, y);
    timer.stop("training");

    std::printf("depth: %lu\n", tree.depth());

    timer.start();
    std::vector<uint32_t> y_pred = tree.predict(X);
    timer.stop("prediction");

    std::printf("accuracy: %.2f\n", accuracy(y_pred, y));

    return 0;
}
