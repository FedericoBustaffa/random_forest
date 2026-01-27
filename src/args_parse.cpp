#include "args_parse.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

Args parse_args(int argc, char** argv)
{
    if (argc < 6)
    {
        std::printf("USAGE: %s <estimators> <max_depth> "
                    "<backend> <threads> <dataset> [log]\n",
                    argv[0]);

        exit(EXIT_FAILURE);
    }

    Args args;

    args.estimators = std::stoull(argv[1]);
    args.max_depth = std::stoull(argv[2]);

    std::string s = argv[3];
    if (s == "seq")
        args.backend = Backend::Sequential;
    else if (s == "omp")
        args.backend = Backend::OpenMP;
    else if (s == "ff")
        args.backend = Backend::FastFlow;
    else if (s == "mpi")
        args.backend = Backend::MPI;

    args.threads = std::stoul(argv[4]);
    if (args.backend == Backend::Sequential)
        args.threads = 1;

    args.nodes = 1;
    if (args.backend == Backend::MPI)
    {
        // initialize MPI if needed
        if (args.backend == Backend::MPI)
            MPI_Init(&argc, &argv);

        int nodes;
        MPI_Comm_size(MPI_COMM_WORLD, &nodes);
        args.nodes = nodes;
    }

    args.dataset = argv[5];
    args.log = false;
    if (argc == 7)
    {
        if (std::strcmp(argv[6], "log") == 0)
            args.log = true;
        else
        {
            std::printf("ERROR: \"%s\" is an invalid value\n", argv[6]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}
