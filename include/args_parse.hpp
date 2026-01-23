#ifndef ARGS_PARSE_HPP
#define ARGS_PARSE_HPP

#include <string>

#include "backend.hpp"

struct Args
{
    // random forest parameters
    size_t estimators;
    size_t max_depth;
    Backend backend;
    size_t threads;
    size_t nodes;

    // dataset used
    std::string dataset;

    // true if want to write a file with results
    bool log;
};

Args parse_args(int argc, char** argv);

#endif
