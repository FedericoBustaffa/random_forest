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
    auto [train_idx, test_idx] = train_test_split(X.size(), 0.2, 42);
    auto [X_train, X_test] = split(X, train_idx, test_idx);
    auto [y_train, y_test] = split(y, train_idx, test_idx);

    DecisionTree tree(max_depth);
    Timer<milli> timer;
    timer.start();
    tree.fit(X_train, y_train);
    timer.stop("training");

    std::printf("size: %lu\n", tree.size());
    std::printf("depth: %lu\n", tree.depth());

    timer.start();
    std::vector<uint32_t> y_pred = tree.predict(X_train);
    timer.stop("train predict");

    std::printf("train accuracy: %.2f\n", accuracy_score(y_pred, y_train));
    std::printf("train f1: %.2f\n", f1_score(y_pred, y_train));

    timer.start();
    y_pred = tree.predict(X_test);
    timer.stop("test predict");

    std::printf("test accuracy: %.2f\n", accuracy_score(y_pred, y_test));
    std::printf("test f1: %.2f\n", f1_score(y_pred, y_test));

    return 0;
}
