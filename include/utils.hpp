#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>

#include "dataframe.hpp"
#include "view.hpp"

DataFrame read_csv(const std::string& filepath);

std::vector<size_t> argsort(const View<double>& v);

double accuracy(const std::vector<unsigned int>& predictions,
                const std::vector<unsigned int>& correct);

#endif
