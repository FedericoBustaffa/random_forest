#ifndef IO_HPP
#define IO_HPP

#include <sstream>
#include <string>

#include "dataframe.hpp"

DataFrame read_csv(const std::string& filepath);

template <typename T>
std::string stringify(const T& obj)
{
    std::stringstream ss;
    ss << obj;
    return ss.str();
}

void print_record(
    const std::vector<std::pair<std::string, std::string>>& record);

void to_json(const std::vector<std::pair<std::string, std::string>>& record);

#endif
