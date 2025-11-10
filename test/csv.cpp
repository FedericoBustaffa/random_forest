#include <iostream>
#include <vector>

#include "csv.hpp"

int main(int argc, const char** argv)
{
    if (argc != 2)
    {
        std::cout << "USAGE: " << argv[0] << " <filepath>" << std::endl;
        return 1;
    }

    std::vector<std::string> headers = {"sepal_length", "sepal_width",
                                        "petal_length", "petal_width", "label"};
    dataframe df = read_csv(argv[1], headers);

    std::cout << "--- info ---" << std::endl;
    std::cout << "shape: (" << df.rows() << ", " << df.columns() << ")"
              << std::endl;

    for (const auto& h : df.headers())
        std::cout << h << ": " << df[h].type() << std::endl;

    auto v = df["label"].as_double();
    for (const auto& i : v)
        std::cout << i << std::endl;

    return 0;
}
