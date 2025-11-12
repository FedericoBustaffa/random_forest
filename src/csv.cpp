#include "csv.hpp"

#include <cassert>
#include <fstream>
#include <sstream>
#include <string>

#include <iostream>

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

bool readline(const std::string& filepath, std::vector<std::string>& line)
{
    std::ifstream file(filepath);
    return readline(file, line);
}

DataFrame read_csv(std::ifstream& file, const std::vector<std::string>& headers)
{
    // the dataframe content
    std::vector<std::string> content;

    // fill the table
    std::vector<std::string> line;
    while (readline(file, line))
    {
        assert(line.size() == headers.size());
        for (size_t i = 0; i < line.size(); i++)
            content.push_back(line[i]);
    }

    return DataFrame(content, headers);
}

DataFrame read_csv(const std::string& filepath,
                   const std::vector<std::string>& headers)
{
    std::ifstream file(filepath);
    return read_csv(file, headers);
}

DataFrame read_csv(const std::string& filepath)
{
    std::ifstream file(filepath);
    std::vector<std::string> headers;
    readline(file, headers);

    return read_csv(file, headers);
}
