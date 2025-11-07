#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "decision_tree.hpp"

int main(int argc, const char** argv)
{
    if (argc != 2)
    {
        std::printf("USAGE: %s <dataset>\n", argv[0]);
        return 1;
    }

    std::ifstream file(argv[1]);
    std::string buffer;
    while (file)
    {
        file >> buffer;
        std::cout << buffer << std::endl;
    }

    return 0;
}
