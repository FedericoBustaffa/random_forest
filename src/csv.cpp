#include "csv.hpp"

#include <fstream>
#include <sstream>
#include <string>

std::vector<std::string> readline(std::ifstream& file)
{
    std::string line;
    if (!std::getline(file, line))
        return {};

    std::stringstream ss;
    ss << line;

    std::string word;
    std::vector<std::string> tokenized_line;
    while (std::getline(ss, word, ','))
        tokenized_line.push_back(word);

    return tokenized_line;
}

std::vector<std::string> readline(const std::string& filepath)
{
    std::ifstream file(filepath);
    return readline(file);
}

dataframe read_csv(std::ifstream& file, const std::vector<std::string>& headers)
{
    // read the first line to get the number of columns
    std::vector<std::string> line = readline(file);

    // vector of columns
    std::vector<std::vector<std::string>> buffer(line.size());

    // fill the first row
    for (size_t i = 0; i < line.size(); i++)
        buffer[i].push_back(line[i]);

    // fill the rest of the table
    do
    {
        line = readline(file);
        if (!line.empty())
            for (size_t i = 0; i < line.size(); i++)
                buffer[i].push_back(line[i]);
    } while (!line.empty());

    return dataframe(buffer, headers);
}

dataframe read_csv(const std::string& filepath,
                   const std::vector<std::string>& headers)
{
    std::ifstream file(filepath);
    return read_csv(file, headers);
}

dataframe read_csv(const std::string& filepath)
{
    std::ifstream file(filepath);
    std::vector<std::string> headers = readline(file);

    return read_csv(file, headers);
}
