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
    for (size_t i = 0; i < headers.size(); i++)
    {
        std::cout << headers[i] << ": " << std::flush;
        if (df.fields()[i] == dataframe::field::numerical)
            std::cout << "numerical" << std::endl;
        if (df.fields()[i] == dataframe::field::categorical)
            std::cout << "categorical" << std::endl;
    }

    // for (const auto& v : df["sepal_length"])
    //     std::cout << v << std::endl;

    std::vector<double> vec = df.to_vector();

    return 0;
}
