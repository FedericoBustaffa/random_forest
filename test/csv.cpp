#include <iostream>

#include <csv.hpp>

using namespace csv;

int main(int argc, const char** argv)
{
    if (argc != 2)
    {
        std::cout << "USAGE: " << argv[0] << " <filepath>" << std::endl;
        return 1;
    }

    CSVReader reader(argv[1]);
    for (const CSVRow& row : reader)
    {
        for (const CSVField& field : row)
            std::cout << field.get_sv() << " " << std::flush;
        std::cout << std::endl;
    }

    for (const auto& cn : reader.get_col_names())
        std::cout << cn << std::endl;

    return 0;
}
