#include "io.hpp"

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

void print_stats(const Record& record)
{
    std::printf("estimators: %lu\n", record.estimators);
    std::printf("max_depth %lu\n", record.max_depth);
    std::printf("backend: %s\n", record.backend.c_str());
    std::printf("threads: %lu\n", record.threads);
    std::printf("nodes: %lu\n", record.nodes);
    std::printf("dataset: %s\n", record.dataset.c_str());
    std::printf("training time: %.2f ms\n", record.train_time);
    std::printf("prediction time: %.2f ms \n", record.predict_time);
    std::printf("accuracy: %.2f\n", record.accuracy);
    std::printf("f1 score: %.2f\n", record.f1);
}

void print_record(const Record& record)
{
    if (record.backend == "mpi")
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0)
            print_stats(record);
    }
    else
        print_stats(record);
}

void write_json(const Record& record)
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
    ss << "result_" << std::setw(0) << std::setfill('0') << nfiles << ".json";

    std::ofstream out(ss.str());

    out << "{\n";
    out << "\t\"estimators\": " << record.estimators << ",\n";
    out << "\t\"max_depth\": " << record.max_depth << ",\n";
    out << "\t\"backend\": " << '\"' << record.backend << '\"' << ",\n";
    out << "\t\"threads\": " << record.threads << ",\n";
    out << "\t\"nodes\": " << record.nodes << ",\n";
    out << "\t\"dataset\": " << '\"' << record.dataset << '\"' << ",\n";
    out << "\t\"accuracy\": " << record.accuracy << ",\n";
    out << "\t\"f1\": " << record.f1 << ",\n";
    out << "\t\"train_time\": " << record.train_time << ",\n";
    out << "\t\"predict_time\": " << record.predict_time << "\n";
    out << "}\n";
}

void to_json(const Record& record)
{
    if (record.backend == "mpi")
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0)
            write_json(record);
    }
    else
        write_json(record);
}
