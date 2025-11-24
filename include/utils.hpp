#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <utility>
#include <vector>

#include "matrix.hpp"
#include "vector.hpp"

std::pair<Matrix, Vector> read_csv(const std::string& filepath,
                                   bool has_headers = false);

std::vector<size_t> argsort(const VectorView& t);

double accuracy(const VectorView& predictions, const VectorView& correct);

#endif
