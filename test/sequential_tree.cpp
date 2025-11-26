#include <chrono>
#include <cstdio>

#include "csv.hpp"
#include "dataframe.hpp"
#include "decision_tree.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc != 2)
    {
        std::printf("USAGE: %s <filepath>\n", argv[0]);
        return 1;
    }

    DataFrame df = read_csv(argv[1]);
    std::vector<std::vector<double>> data = df.toVector();
    std::vector<std::vector<double>> X(data.begin(), data.end() - 1);
    std::vector<uint32_t> y;
    for (const auto& i : data.back())
        y.push_back(i);

    DecisionTree tree;
    auto start = std::chrono::high_resolution_clock::now();
    tree.fit(X, y);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::printf("trained in %.4f seconds\n", duration.count());

    std::vector<uint32_t> y_pred = tree.predict(X);
    std::printf("accuracy: %.2f\n", accuracy(y_pred, y));

    return 0;
}
