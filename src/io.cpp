#include "io.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mpi.h>
#include <sstream>
#include <string>

#include "dataframe.hpp"

namespace fs = std::filesystem;

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

void print_fields(
    const std::vector<std::pair<std::string, std::string>>& record)
{
    for (const auto& field : record)
        std::printf("%s: %s\n", field.first.c_str(), field.second.c_str());
}

void print_record(
    const std::vector<std::pair<std::string, std::string>>& record)
{
    auto it = std::find_if(record.begin(), record.end(),
                           [](const auto& p) { return p.first == "backend"; });

    if (it != record.end() && it->second == "mpi")
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0)
            print_fields(record);
    }
    else
        print_fields(record);
}

bool is_number(const std::string& s)
{
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end && *end == '\0';
}

void write_json(const std::vector<std::pair<std::string, std::string>>& record)
{
    fs::path dir_path = "tmp";
    if (!fs::exists(dir_path))
        fs::create_directory(dir_path);

    size_t nfiles = 0;
    for (const auto& f : fs::directory_iterator(dir_path))
    {
        std::string name = f.path().filename();
        nfiles++;
    }

    std::stringstream ss;
    ss << dir_path.c_str() << "/";
    ss << "result_" << nfiles << ".json";

    std::ofstream out(ss.str());
    out << "{\n";

    size_t n = record.size();
    size_t i = 0;
    for (const auto& field : record)
    {
        if (is_number(field.second))
            out << "    \"" << field.first << "\": " << field.second;
        else
            out << "    \"" << field.first << "\": \"" << field.second << "\"";

        if (i != n - 1)
            out << ",";

        out << "\n";
        i++;
    }
    out << "}\n";
}

void to_json(const std::vector<std::pair<std::string, std::string>>& record)
{
    auto it = std::find_if(record.begin(), record.end(),
                           [](const auto& p) { return p.first == "backend"; });

    if (it != record.end() && it->second == "mpi")
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0)
            write_json(record);
    }
    else
    {
        write_json(record);
    }
}
