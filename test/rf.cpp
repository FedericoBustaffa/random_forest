#include <mpi.h>

#include "args_parse.hpp"
#include "dataframe.hpp"
#include "datasplit.hpp"
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
    DataSplit data = train_test_split(X, y, 0.2, 42);

    // initialize MPI if needed
    if (args.backend == Backend::MPI)
        MPI_Init(&argc, &argv);

    // to store statistics

    // to measure performance
    Timer<milli> timer;

    RandomForest forest(args.estimators, args.max_depth, args.backend,
                        args.threads);
    timer.start();
    forest.fit(data);
    float train_time = timer.stop();

    timer.start();
    std::vector<uint8_t> pred = forest.predict(data.X_test);
    float predict_time = timer.stop();

    // prediction scores
    float accuracy = accuracy_score(pred, data.y_test);
    float f1 = f1_score(pred, data.y_test);

    Record record(args);
    record.nodes = forest.nodes();
    record.train_time = train_time;
    record.predict_time = predict_time;
    record.accuracy = accuracy;
    record.f1 = f1;

    print_record(record);

    // save statistics on a json file
    if (args.log)
        to_json(record);

    // Finalize MPI if needed
    if (args.backend == Backend::MPI)
        MPI_Finalize();

    return 0;
}
