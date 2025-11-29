#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "random_forest.hpp"

struct Record
{
    std::string dataset;
    std::string backend;
    size_t estimators;
    size_t max_depth;
    double accuracy;
    double train_time;
    double predict_time;
    size_t threads;
    size_t nodes;
};

std::vector<size_t> argsort(const std::vector<double>& v,
                            const std::vector<size_t>& indices);

std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>>& X);

std::unordered_map<uint32_t, size_t> count(const std::vector<uint32_t>& y,
                                           const std::vector<size_t>& indices);

uint32_t majority(const std::vector<uint32_t>& y, std::vector<size_t>& indices);

std::vector<size_t> bootstrap(size_t n_samples);

double accuracy_score(const std::vector<uint32_t>& predictions,
                      const std::vector<uint32_t>& correct);

Backend to_backend(const std::string& s);

void to_json(const Record& record);

#endif
