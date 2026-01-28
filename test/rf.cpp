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

    // to store statistics

    // to measure performance
    Timer<milli> timer;

    RandomForest forest(args.estimators, args.max_depth, args.backend,
                        args.threads);
    timer.start();
    forest.fit(data.X_train, data.y_train);
    float train_time = timer.stop();

    timer.start();
    std::vector<uint8_t> pred = forest.predict(data.X_test);
    float predict_time = timer.stop();

    // prediction scores
    float accuracy = accuracy_score(pred, data.y_test);
    float f1 = f1_score(pred, data.y_test);

    std::vector<std::pair<std::string, std::string>> record;
    record.emplace_back("dataset", args.dataset);
    record.emplace_back("estimators", stringify(args.estimators));
    record.emplace_back("max_depth", stringify(args.max_depth));
    record.emplace_back("accuracy", stringify(accuracy));
    record.emplace_back("f1", stringify(f1));
    record.emplace_back("backend", argv[3]);
    record.emplace_back("nodes", stringify(args.nodes));
    record.emplace_back("threads", stringify(args.threads));
    record.emplace_back("train_time", stringify(train_time));
    record.emplace_back("predict_time", stringify(predict_time));

    print_record(record);

    // save statistics on a json file
    if (args.log)
        to_json(record);

    // Finalize MPI if needed
    if (args.backend == Backend::MPI)
        MPI_Finalize();

    return 0;
}
