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
    Timer timer;
    timer.start();
    forest.fit(X, y);
    double train_time = timer.stop("training");

    timer.start();
    std::vector<uint32_t> y_pred = forest.predict(X);
    double predict_time = timer.stop("prediction");

    double accuracy = accuracy_score(y_pred, y);
    std::printf("accuracy: %.2f\n", accuracy);

    Record record;
    record.dataset = argv[5];
    record.backend = argv[3];
    record.estimators = estimators;
    record.max_depth = max_depth;
    record.accuracy = accuracy;
    record.train_time = train_time;
    record.predict_time = predict_time;

    switch (backend)
    {
    case Backend::Sequential:
        record.threads = 1;
        record.nodes = 1;
        break;

    case Backend::OpenMP:
        record.threads = n_threads;
        record.nodes = 1;
        break;

    case Backend::FastFlow:
        record.threads = n_threads;
        record.nodes = 1;
        break;

    default:
        break;
    }

    to_json(record);

    return 0;
}
