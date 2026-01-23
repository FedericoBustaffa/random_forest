#include <mpi.h>

#include "args_parse.hpp"
#include "dataframe.hpp"
#include "io.hpp"
#include "metrics.hpp"
#include "random_forest.hpp"
#include "timer.hpp"
#include "utils.hpp"

int main(int argc, char** argv)
{
    // CLI args parsing
    Args args = parse_args(argc, argv);

    // get the dataset and split train and test sets
    DataFrame df = read_csv(args.dataset);
    auto [X, y] = df.to_vector();
    auto [train_idx, test_idx] = train_test_split(X.size(), 0.2, 42);
    auto [X_train, X_test] = split(X, train_idx, test_idx);
    auto [y_train, y_test] = split(y, train_idx, test_idx);

    // initialize MPI if needed
    if (args.backend == Backend::MPI)
        MPI_Init(&argc, &argv);

    // to store statistics
    Record record(args);

    // to measure performance
    Timer<milli> timer;

    RandomForest forest(args.estimators, args.max_depth, args.backend,
                        args.threads, args.nodes);
    timer.start();
    forest.fit(X_train, y_train);
    record.train_time = timer.stop();

    timer.start();
    std::vector<uint8_t> train_pred = forest.predict(X_train);
    record.train_predict_time = timer.stop();

    timer.start();
    std::vector<uint8_t> test_pred = forest.predict(X_test);
    record.test_predict_time = timer.stop();

    // prediction scores
    record.train_accuracy = accuracy_score(train_pred, y_train);
    record.train_f1 = f1_score(train_pred, y_train);
    record.test_accuracy = accuracy_score(test_pred, y_test);
    record.test_f1 = f1_score(test_pred, y_test);

    print_record(record);

    // save statistics on a json file
    if (args.log)
        to_json(record);

    // Finalize MPI if needed
    if (args.backend == Backend::MPI)
        MPI_Finalize();

    return 0;
}
