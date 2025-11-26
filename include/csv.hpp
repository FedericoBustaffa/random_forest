#ifndef CSV_HPP
#define CSV_HPP

#include <string>

#include "dataframe.hpp"

DataFrame read_csv(const std::string& filepath);

#endif
