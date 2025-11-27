#include <chrono>
#include <cstdio>

#include "csv.hpp"
#include "dataframe.hpp"
#include "random_forest.hpp"
#include "random_forest_omp.hpp"
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
    RandomForestOMP forest_omp(estimators, max_depth);

    auto start = std::chrono::high_resolution_clock::now();
    forest.fit(X, y);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double sequential_time = duration.count();
    std::printf("sequential training time: %.4f seconds\n", sequential_time);

    start = std::chrono::high_resolution_clock::now();
    forest_omp.fit(X, y);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    double omp_time = duration.count();
    std::printf("sequential training time: %.4f seconds\n", omp_time);

    std::printf("speedup: %.4f\n", sequential_time / omp_time);

    std::vector<uint32_t> y_pred = forest.predict(X);
    std::printf("accuracy: %.2f\n", accuracy(y_pred, y));

    return 0;
}
