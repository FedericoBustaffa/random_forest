#include <cstdio>
#include <mpi.h>

#include "csv.hpp"
#include "dataframe.hpp"
#include "random_forest.hpp"
#include "timer.hpp"
#include "utils.hpp"

void exit_with_msg(const char* program)
{
    std::printf("USAGE: %s <estimators> <max_depth> <filepath> <log> <backend> "
                "<threads> <nodes>\n",
                program);
    exit(1);
}

Record parse(int argc, char** argv, bool& log)
{
    if (argc < 5)
        exit_with_msg(argv[0]);

    Record record;

    record.estimators = std::stoull(argv[1]);
    record.max_depth = std::stoull(argv[2]);
    record.dataset = argv[3];
    log = std::stoi(argv[4]);

    if (argc == 5)
    {
        record.backend = to_backend("seq");
        record.threads = 1;
        record.nodes = 1;

        return record;
    }

    if (argc > 5)
    {
        record.backend = to_backend(argv[5]);
        if (record.backend == Backend::Invalid)
        {
            std::printf("[ERROR]: %s is an invalid backend", argv[4]);
            exit(1);
        }

        if (record.backend == Backend::Sequential)
        {
            record.threads = 1;
            record.nodes = 1;
        }
        else if (record.backend != Backend::MPI)
        {
            record.threads = std::stoul(argv[6]);
            record.nodes = 1;
        }
        else
        {
            record.threads = std::stoul(argv[6]);
            record.nodes = std::stoul(argv[7]);
        }
    }

    return record;
}

int main(int argc, char** argv)
{
    // CLI args parsing
    bool log;
    Record record = parse(argc, argv, log);

    // get the dataset
    DataFrame df = read_csv(record.dataset);
    auto [X, y] = df.to_vector();
    auto [train_idx, test_idx] = train_test_split(X.size(), 0.2);
    auto [X_train, X_test] = split(X, train_idx, test_idx);
    auto [y_train, y_test] = split(y, train_idx, test_idx);

    // initialize MPI if needed
    if (record.backend == Backend::MPI)
        MPI_Init(&argc, &argv);

    RandomForest forest(record.estimators, record.max_depth, record.backend,
                        record.threads, record.nodes);
    Timer<milli> timer;
    timer.start();
    forest.fit(X_train, y_train);
    double train_time = timer.stop("training");

    timer.start();
    std::vector<uint32_t> train_pred = forest.predict(X_train);
    double train_predict_time = timer.stop("train prediction");

    timer.start();
    std::vector<uint32_t> test_pred = forest.predict(X_test);
    timer.stop("test prediction");

    double train_accuracy = accuracy_score(train_pred, y_train);
    std::printf("train accuracy: %.2f\n", train_accuracy);

    double test_accuracy = accuracy_score(test_pred, y_test);
    std::printf("test accuracy: %.2f\n", test_accuracy);

    if (log)
    {
        record.accuracy = train_accuracy;
        record.train_time = train_time;
        record.predict_time = train_predict_time;

        to_json(record);
    }

    if (record.backend == Backend::MPI)
        MPI_Finalize();

    return 0;
}
