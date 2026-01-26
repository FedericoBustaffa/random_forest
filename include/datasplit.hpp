#ifndef DATASPLIT_HPP
#define DATASPLIT_HPP

#include <cstdint>
#include <vector>

enum class FeatureType
{
    Binary,
    Continuous
};

struct DataSplit
{
    DataSplit(const std::vector<std::vector<float>>& X_train,
              const std::vector<std::vector<float>>& X_test,
              const std::vector<uint8_t>& y_train,
              const std::vector<uint8_t>& y_test)
        : X_train(std::move(X_train)), X_test(std::move(X_test)),
          y_train(std::move(y_train)), y_test(std::move(y_test))
    {
        size_t n_features = this->X_train[0].size();
        size_t n_samples = this->X_train.size();
        feature_types.reserve(n_features);

        for (size_t i = 0; i < n_features; ++i)
        {
            feature_types.push_back(FeatureType::Binary);
            for (size_t j = 0; j < n_samples; ++j)
            {
                if (this->X_train[j][i] != 0 && this->X_train[j][i] != 1)
                {
                    feature_types[i] = FeatureType::Continuous;
                    break;
                }
            }
        }
    }

    ~DataSplit() {}

    std::vector<std::vector<float>> X_train, X_test;
    std::vector<uint8_t> y_train, y_test;
    std::vector<FeatureType> feature_types;
};

#endif
