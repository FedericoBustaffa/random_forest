#include "utils.hpp"

#include <algorithm>
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

double accuracy_score(const std::vector<unsigned int>& predictions,
                      const std::vector<unsigned int>& correct)
{
    double counter = 0.0;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == correct[i])
            counter++;
    }

    return counter / predictions.size();
}

void to_json(const char* prefix, size_t estimators, size_t max_depth,
             double train_time, double predict_time, double accuracy,
             int nthreads)
{
    fs::path dir_path = "results";
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
    ss << "result_" << std::setw(3) << std::setfill('0') << nfiles << ".json";

    std::ofstream out(ss.str());
    out << "{\n";
    out << "\t\"threading\": " << '\"' << prefix << '\"' << ",\n";
    out << "\t\"estimators\": " << estimators << ",\n";
    out << "\t\"max_depth\": " << max_depth << ",\n";
    out << "\t\"train_time\": " << train_time << ",\n";
    out << "\t\"predict_time\": " << predict_time << ",\n";
    out << "\t\"accuracy\": " << accuracy << ",\n";
    out << "\t\"nthreads\": " << nthreads << "\n";
    out << "}\n";
}
