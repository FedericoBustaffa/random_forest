#include <cstdio>

#include "decision_tree.hpp"
#include "utils.hpp"

int main(int argc, const char** argv)
{
    if (argc != 2)
    {
        std::printf("USAGE: %s <filepath>\n", argv[0]);
        return 1;
    }

    auto [X, y] = read_csv(argv[1]);

    DecisionTree tree;
    tree.fit(X, y);
    // Tensor predictions = tree.predict(X);
    std::printf("accuracy: %.2f\n", accuracy(y, y));

    return 0;
}
