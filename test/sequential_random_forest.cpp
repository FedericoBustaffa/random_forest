#include <chrono>
#include <cstdio>

#include "dataframe.hpp"
#include "random_forest.hpp"
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

    RandomForest forest(100);
    auto start = std::chrono::high_resolution_clock::now();
    forest.fit(X, y);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::printf("trained in %.4f seconds\n", duration.count());

    for (const auto& d : forest.depths())
        std::printf("depth: %lu\n", d);

    std::vector<uint32_t> y_pred = forest.predict(X);
    // std::printf("accuracy: %.2f\n", accuracy(y_pred, y));

    return 0;
}
