#ifndef CSV_HPP
#define CSV_HPP

#include <string>

#include "args_parse.hpp"
#include "backend.hpp"
#include "dataframe.hpp"

struct Record
{
    Record(const Args& args)
        : estimators(args.estimators), max_depth(args.max_depth),
          threads(args.threads), nodes(args.nodes), dataset(args.dataset)

    {
        switch (args.backend)
        {
        case Backend::Sequential:
            backend = "seq";
            break;

        case Backend::OpenMP:
            backend = "omp";
            break;

        case Backend::FastFlow:
            backend = "ff";
            break;

        case Backend::MPI:
            backend = "mpi";
            break;

        default:
            break;
        }
    }

    // random forest parameters
    size_t estimators;
    size_t max_depth;
    std::string backend;
    size_t threads;
    size_t nodes;

    // dataset used
    std::string dataset;

    // predictions results
    float train_accuracy;
    float test_accuracy;
    float train_f1;
    float test_f1;

    // performance metrics
    float train_time;
    float train_predict_time;
    float test_predict_time;
};

DataFrame read_csv(const std::string& filepath);

void print_record(const Record& record);

void to_json(const Record& record);

#endif
