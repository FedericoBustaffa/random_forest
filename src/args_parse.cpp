#include "utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "args_parse.hpp"

Args parse_args(int argc, char** argv)
{
    if (argc < 7)
    {
        std::printf("USAGE: %s <estimators> <max_depth> "
                    "<backend> <threads> <nodes> <dataset> [log]\n",
                    argv[0]);

        exit(EXIT_FAILURE);
    }

    Args args;

    args.estimators = std::stoull(argv[1]);
    args.max_depth = std::stoull(argv[2]);
    args.backend = to_backend(argv[3]);
    args.threads = std::stoul(argv[4]);
    args.nodes = std::stoul(argv[5]);
    args.dataset = argv[6];
    args.log = false;

    if (argc == 8)
    {
        if (std::strcmp(argv[7], "log") == 0)
            args.log = true;
        else
        {
            std::printf("ERROR: \"%s\" is an invalid value\n", argv[7]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}
