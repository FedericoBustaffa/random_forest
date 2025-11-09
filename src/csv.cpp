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
    std::vector<std::string> buffer; // to build the dataframe
    std::vector<std::string> line;
    while (true)
    {
        line = readline(file);
        if (line.empty())
            break;

        for (const auto& word : line)
            buffer.push_back(word);
    }

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
