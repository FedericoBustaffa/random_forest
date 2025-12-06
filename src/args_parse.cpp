#include "utils.hpp"

#include <cstdio>
#include <cstdlib>

#include "args_parse.hpp"

Args parse_args(int argc, char** argv)
{
    if (argc != 8)
    {
        std::printf("USAGE: %s <estimators> <max_depth> "
                    "<backend> <threads> <nodes> "
                    "<dataset> <log>\n",
                    argv[0]);

        exit(1);
    }

    Args args;

    args.estimators = std::stoull(argv[1]);
    args.max_depth = std::stoull(argv[2]);
    args.backend = to_backend(argv[3]);
    args.threads = std::stoul(argv[4]);
    args.nodes = std::stoul(argv[5]);
    args.dataset = argv[6];
    args.log = (bool)std::stoi(argv[7]);

    return args;
}
