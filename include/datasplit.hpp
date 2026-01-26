#ifndef DATASPLIT_HPP
#define DATASPLIT_HPP

#include <cstdint>
#include <vector>

struct DataSplit
{
    DataSplit(const std::vector<std::vector<float>>& X_train,
              const std::vector<std::vector<float>>& X_test,
              const std::vector<uint8_t>& y_train,
              const std::vector<uint8_t>& y_test)
        : X_train(std::move(X_train)), X_test(std::move(X_test)),
          y_train(std::move(y_train)), y_test(std::move(y_test))
    {
    }

    ~DataSplit() {}

    std::vector<std::vector<float>> X_train, X_test;
    std::vector<uint8_t> y_train, y_test;
};

#endif
