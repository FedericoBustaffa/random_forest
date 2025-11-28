#include <chrono>
#include <cstdio>

#include "csv.hpp"
#include "dataframe.hpp"
#include "decision_tree.hpp"
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
    auto [X, y] = df.toVector();

    DecisionTree tree(max_depth);
    auto start = std::chrono::high_resolution_clock::now();
    tree.fit(X, y);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::printf("training time: %.4f seconds\n", duration.count());

    std::printf("depth: %lu\n", tree.depth());

    start = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> y_pred = tree.predict(X);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::printf("prediction time: %.4f seconds\n", duration.count());

    std::printf("accuracy: %.2f\n", accuracy(y_pred, y));

    return 0;
}
