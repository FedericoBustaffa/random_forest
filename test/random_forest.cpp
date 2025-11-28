#include <cstdio>

#include "csv.hpp"
#include "dataframe.hpp"
#include "random_forest.hpp"
#include "timer.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc != 4)
    {
        std::printf("USAGE: %s <estimators> <max_depth> <filepath>\n", argv[0]);
        return 1;
    }

    size_t estimators = std::stoull(argv[1]);
    size_t max_depth = std::stoull(argv[2]);

    DataFrame df = read_csv(argv[3]);
    auto [X, y] = df.toVector();

    RandomForest forest(estimators, max_depth);
    Timer<milli> timer;
    timer.start();
    forest.fit(X, y);
    timer.stop("training");

    timer.start();
    std::vector<uint32_t> y_pred = forest.predict(X);
    timer.stop("prediction");

    std::printf("accuracy: %.2f\n", accuracy(y_pred, y));

    return 0;
}
