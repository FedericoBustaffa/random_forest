#include <cstdio>
#include <omp.h>

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
    auto [X, y] = df.to_vector();

    RandomForest forest(estimators, max_depth);
    Timer<milli> timer;
    timer.start();
    forest.fit(X, y);
    double train_time = timer.stop("training");

    timer.start();
    std::vector<uint32_t> y_pred = forest.predict(X);
    double prediction_time = timer.stop("prediction");

    double accuracy = accuracy_score(y_pred, y);
    std::printf("accuracy: %.2f\n", accuracy);

    to_json("omp", estimators, max_depth, train_time, prediction_time, accuracy,
            omp_get_max_threads());

    return 0;
}
