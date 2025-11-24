#include "utils.hpp"

#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

enum class DataType
{
    Numerical,
    Categorical
};

bool readline(std::ifstream& file, std::vector<std::string>& line)
{
    std::string stringline;
    if (!std::getline(file, stringline) || stringline.empty())
        return false;

    std::stringstream ss;
    ss << stringline;

    line.clear();
    std::string word;
    while (std::getline(ss, word, ','))
        line.push_back(word);

    return true;
}

std::pair<Matrix, Vector> read_csv(const std::string& filepath,
                                   bool has_headers)
{
    std::ifstream file(filepath);

    // just read headers and discard them if present
    std::vector<std::string> line;
    if (has_headers)
        readline(file, line);

    // read the first line to check which column needs numerical conversion
    readline(file, line);
    std::vector<DataType> datatypes;
    std::unordered_map<size_t, std::unordered_map<std::string, double>>
        dictionaries;

    // columns counter
    size_t cols = line.size();

    // regex to capture every possible numerical type
    std::regex numerical(
        R"(^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$)");

    size_t rows = 1;              // rows counter;
    std::vector<double> features; // assume first n-1 columns are features
    std::vector<double> targets;  // assume the last column as targets

    // setup dictionaries for on the fly conversion and insert first line
    for (size_t i = 0; i < line.size() - 1; i++)
    {
        // infer columns types
        if (std::regex_search(line[i], numerical))
        {
            datatypes.push_back(DataType::Numerical);
            features.push_back(std::stod(line[i]));
        }
        else
        {
            datatypes.push_back(DataType::Categorical);
            dictionaries[i] = {};
            dictionaries[i][line[i]] = 0.0;
            features.push_back(dictionaries[i][line[i]]);
        }
    }

    // infer target type
    if (std::regex_search(line[cols - 1], numerical))
    {
        datatypes.push_back(DataType::Numerical);
        targets.push_back(std::stod(line[cols - 1]));
    }
    else
    {
        datatypes.push_back(DataType::Categorical);
        dictionaries[cols - 1] = {};
        dictionaries[cols - 1][line[cols - 1]] = 0.0;
        targets.push_back(dictionaries[cols - 1][line[cols - 1]]);
    }

    // read and convert the file on the file
    // in the end it retains only a numerical copy of the CSV
    while (readline(file, line))
    {
        for (size_t i = 0; i < line.size() - 1; i++)
        {
            if (datatypes[i] == DataType::Numerical)
                features.push_back(std::stod(line[i]));
            else
            {
                if (!dictionaries[i].contains(line[i]))
                    dictionaries[i][line[i]] = dictionaries[i].size();

                features.push_back(dictionaries[i][line[i]]);
            }
        }

        if (datatypes[cols - 1] == DataType::Numerical)
            targets.push_back(std::stod(line[cols - 1]));
        else
        {
            if (!dictionaries[cols - 1].contains(line[cols - 1]))
                dictionaries[cols - 1][line[cols - 1]] =
                    dictionaries[cols - 1].size();

            targets.push_back(dictionaries[cols - 1][line[cols - 1]]);
        }
        rows++;
    }

    return {Matrix(features.data(), rows, cols - 1),
            Vector(targets.data(), rows)};
}
