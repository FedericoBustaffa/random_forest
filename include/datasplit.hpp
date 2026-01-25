#ifndef DATASPLIT_HPP
#define DATASPLIT_HPP

#include <cstdint>
#include <vector>

struct DataSplit
{
    DataSplit(size_t train_size, size_t test_size)
        : X_train(train_size), X_test(test_size), y_train(train_size),
          y_test(test_size)
    {
    }

    ~DataSplit() {}

    std::vector<std::vector<float>> X_train, X_test;
    std::vector<uint8_t> y_train, y_test;
};

#endif
