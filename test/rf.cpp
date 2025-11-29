#include <cstdio>

#include "csv.hpp"
#include "dataframe.hpp"
#include "random_forest.hpp"
#include "timer.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc != 6)
    {
        std::printf("USAGE: %s <estimators> <max_depth> <backend> <n_threads> "
                    "<filepath>\n",
                    argv[0]);
        return 1;
    }

    size_t estimators = std::stoull(argv[1]);
    size_t max_depth = std::stoull(argv[2]);

    Backend backend = to_backend(argv[3]);
    if (backend == Backend::Invalid)
    {
        std::printf("[ERROR]: %s is an invalid backend", argv[3]);
        return 1;
    }
    size_t n_threads = std::stoul(argv[4]);

    DataFrame df = read_csv(argv[5]);
    auto [X, y] = df.to_vector();

    RandomForest forest(estimators, max_depth, backend, n_threads);
    Timer<milli> timer;
    timer.start();
    forest.fit(X, y);
    timer.stop("training");

    timer.start();
    std::vector<uint32_t> y_pred = forest.predict(X);

    timer.stop("prediction");

    double accuracy = accuracy_score(y_pred, y);
    std::printf("accuracy: %.2f\n", accuracy);

    return 0;
}
