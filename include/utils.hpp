#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <utility>

#include "tensor.hpp"

std::pair<Tensor, Tensor> read_csv(const std::string& filepath,
                                   bool has_headers = false);

double accuracy(const Tensor& predictions, const Tensor& correct);

#endif
