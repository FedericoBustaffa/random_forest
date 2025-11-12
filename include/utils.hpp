#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

#include "dataframe.hpp"

std::vector<double> encode(const DataFrame* df, size_t col);

std::vector<double> convert(const DataFrame* df, size_t col);

std::vector<double> slice(const std::vector<double>& v, size_t start,
                          size_t stop);

std::vector<double> slice(const std::vector<double>& v, size_t index);

double accuracy(std::vector<double>& guessed, std::vector<double>& correct);

#endif
