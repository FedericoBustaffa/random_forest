#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

namespace fs = std::filesystem;

std::vector<size_t> argsort(const std::vector<double>& v,
                            const std::vector<size_t>& indices)
{
    std::vector<size_t> order(indices.size());
    std::iota(order.begin(), order.end(), 0);

    auto compare = [&](const auto& a, const auto& b) {
        return v[indices[a]] < v[indices[b]];
    };
    std::sort(order.begin(), order.end(), compare);

    return order;
}

std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>>& X)
{
    size_t rows = X.size();
    size_t cols = X[0].size();

    std::vector<std::vector<double>> T(X[0].size());
    for (size_t i = 0; i < cols; i++)
        T[i].reserve(rows);

    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            T[j].push_back(X[i][j]);

    return T;
}

std::unordered_map<uint32_t, size_t> count(const std::vector<uint32_t>& y,
                                           const std::vector<size_t>& indices)
{
    std::unordered_map<uint32_t, size_t> counter;
    for (size_t i = 0; i < indices.size(); i++)
        counter[y[indices[i]]]++;

    return counter;
}

uint32_t majority(const std::vector<uint32_t>& y, std::vector<size_t>& indices)
{
    std::unordered_map<uint32_t, size_t> counter = count(y, indices);
    uint32_t value = 0;
    size_t best_counter = 0;
    for (const auto& kv : counter)
    {
        if (kv.second > best_counter)
        {
            best_counter = kv.second;
            value = kv.first;
        }
    }

    return value;
}

std::pair<std::vector<size_t>, std::vector<size_t>> train_test_split(
    size_t n_samples, float test_size)
{
    std::random_device rd;
    std::mt19937 engine(rd());

    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), engine);

    std::vector<size_t> train_indices(indices.begin(),
                                      indices.begin() + n_samples * test_size);

    std::vector<size_t> test_indices(indices.begin() + n_samples * test_size,
                                     indices.end());

    return {train_indices, test_indices};
}

std::vector<size_t> bootstrap(size_t n_samples)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(0, n_samples - 1);

    std::vector<size_t> indices(n_samples);
    for (size_t i = 0; i < n_samples; i++)
        indices[i] = dist(rng);

    return indices;
}

double accuracy_score(const std::vector<uint32_t>& predictions,
                      const std::vector<uint32_t>& correct)
{
    assert(predictions.size() == correct.size());
    double counter = 0.0;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == correct[i])
            counter++;
    }

    return counter / predictions.size();
}

Backend to_backend(const std::string& s)
{
    if (s == "seq")
        return Backend::Sequential;

    if (s == "omp")
        return Backend::OpenMP;

    if (s == "ff")
        return Backend::FastFlow;

    if (s == "mpi")
        return Backend::MPI;

    return Backend::Invalid;
}

std::string to_string(const Backend& backend)
{
    switch (backend)
    {
    case Backend::Sequential:
        return "seq";

    case Backend::OpenMP:
        return "omp";

    case Backend::FastFlow:
        return "ff";

    case Backend::MPI:
        return "mpi";

    default:
        return "";
    }
}

void to_json(const Record& record)
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
    out << "\t\"dataset\": " << '\"' << record.dataset << '\"' << ",\n";
    out << "\t\"backend\": " << '\"' << to_string(record.backend) << '\"'
        << ",\n";
    out << "\t\"estimators\": " << record.estimators << ",\n";
    out << "\t\"max_depth\": " << record.max_depth << ",\n";
    out << "\t\"accuracy\": " << record.accuracy << ",\n";
    out << "\t\"train_time\": " << record.train_time << ",\n";
    out << "\t\"predict_time\": " << record.predict_time << ",\n";
    out << "\t\"threads\": " << record.threads << ",\n";
    out << "\t\"nodes\": " << record.nodes << "\n";
    out << "}\n";
}
