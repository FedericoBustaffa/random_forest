#include <fstream>
#include <sstream>
#include <string>

#include "dataframe.hpp"

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

DataFrame read_csv(const std::string& filepath)
{
    std::ifstream file(filepath);

    std::vector<std::string> content;
    std::vector<std::string> line;
    readline(file, line);

    for (size_t i = 0; i < line.size(); i++)
        content.push_back(line[i]);

    // columns counter
    size_t cols = line.size();
    size_t rows = 1;

    while (readline(file, line))
    {
        for (size_t i = 0; i < line.size(); i++)
            content.push_back(line[i]);
        rows++;
    }

    return DataFrame(content, rows, cols);
}
