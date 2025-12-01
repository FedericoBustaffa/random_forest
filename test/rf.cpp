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
    bool log;
    Record record = parse(argc, argv, log);
    DataFrame df = read_csv(record.dataset);

    auto [X, y] = df.to_vector();

    if (record.backend == Backend::MPI)
        MPI_Init(&argc, &argv);

    RandomForest forest(record.estimators, record.max_depth, record.backend,
                        record.threads, record.nodes);
    Timer<milli> timer;
    timer.start();
    forest.fit(X, y);
    double train_time = timer.stop("training");

    timer.start();
    std::vector<uint32_t> y_pred = forest.predict(X);
    double predict_time = timer.stop("prediction");

    double accuracy = accuracy_score(y_pred, y);
    std::printf("accuracy: %.2f\n", accuracy);

    if (log)
    {
        record.accuracy = accuracy;
        record.train_time = train_time;
        record.predict_time = predict_time;

        to_json(record);
    }

    if (record.backend == Backend::MPI)
        MPI_Finalize();

    return 0;
}
