#include <cstdio>
#include <cstring>

#include "dataframe.hpp"
#include "decision_tree.hpp"
#include "io.hpp"
#include "metrics.hpp"
#include "timer.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc < 3)
    {
        std::printf("USAGE: %s <max_depth> <filepath> [log]\n", argv[0]);
        return 1;
    }

    bool log = false;
    if (argc == 4)
    {
        if (std::strcmp(argv[3], "log") == 0)
            log = true;
        else
        {
            std::printf("ERROR: \"%s\" is an invalid value\n", argv[6]);
            exit(EXIT_FAILURE);
        }
    }

    size_t max_depth = std::stoull(argv[1]);

    DataFrame df = read_csv(argv[2]);
    auto [X, y] = df.to_vector();
    DataSplit data = train_test_split(X, y, 0.2, 42);

    DecisionTree tree(max_depth);
    Timer<milli> timer;
    timer.start();
    tree.fit(data.X_train, data.y_train);
    double train_time = timer.stop();

    timer.start();
    std::vector<uint8_t> y_pred = tree.predict(data.X_test);
    double predict_time = timer.stop();

    double accuracy = accuracy_score(y_pred, data.y_test);
    double f1 = f1_score(y_pred, data.y_test);

    std::vector<std::pair<std::string, std::string>> record;
    record.emplace_back("implementation", "proposed");
    record.emplace_back("depth", stringify(tree.depth()));
    record.emplace_back("dataset", argv[2]);
    record.emplace_back("accuracy", stringify(accuracy));
    record.emplace_back("f1", stringify(f1));
    record.emplace_back("train_time", stringify(train_time));
    record.emplace_back("predict_time", stringify(predict_time));

    print_record(record);

    // save statistics on a json file
    if (log)
        to_json(record);

    return 0;
}
