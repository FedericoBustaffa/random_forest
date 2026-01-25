#include <cstdio>

#include "dataframe.hpp"
#include "decision_tree.hpp"
#include "io.hpp"
#include "metrics.hpp"
#include "timer.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc != 3)
    {
        std::printf("USAGE: %s <max_depth> <filepath>\n", argv[0]);
        return 1;
    }

    size_t max_depth = std::stoull(argv[1]);

    DataFrame df = read_csv(argv[2]);
    auto [X, y] = df.to_vector();
    DataSplit data = train_test_split(X, y, 0.2, 42);

    DecisionTree tree(max_depth);
    Timer<milli> timer;
    timer.start();
    tree.fit(data.X_train, data.y_train);
    timer.stop("traininig");

    timer.start();
    std::vector<uint8_t> y_pred = tree.predict(data.X_test);
    timer.stop("prediction");

    std::printf("accuracy: %.2f\n", accuracy_score(y_pred, data.y_test));
    std::printf("f1 score: %.2f\n", f1_score(y_pred, data.y_test));

    std::printf("size: %lu\n", tree.size());
    std::printf("depth: %lu\n", tree.depth());

    return 0;
}
