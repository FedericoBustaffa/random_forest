#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "backend.hpp"
#include "counter.hpp"
#include "datasplit.hpp"

std::vector<size_t> argsort(const std::vector<float>& v,
                            const std::vector<size_t>& indices);

std::vector<std::vector<float>> transpose(
    const std::vector<std::vector<float>>& X);

size_t count_labels(const std::vector<uint8_t>& y);

size_t count_labels(const std::vector<uint8_t>& y,
                    const std::vector<size_t>& indices);

Counter count(const std::vector<uint8_t>& y,
              const std::vector<size_t>& indices);

uint8_t majority(const std::vector<uint8_t>& y,
                 const std::vector<size_t>& indices);

DataSplit train_test_split(const std::vector<std::vector<float>>& X,
                           const std::vector<uint8_t>& y, float test_size,
                           int seed = -1);

std::vector<size_t> bootstrap(size_t n_samples, uint8_t seed);

Backend to_backend(const std::string& s);

#endif
