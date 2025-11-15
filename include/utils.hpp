#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <utility>
#include <vector>

#include "tensor.hpp"

std::pair<Tensor, Tensor> read_csv(const std::string& filepath,
                                   bool has_headers = false);

std::vector<size_t> argsort(const TensorView& t);

double accuracy(const TensorView& predictions, const TensorView& correct);

#endif
