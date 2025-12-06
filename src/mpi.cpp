#include "random_forest.hpp"

#include <cstdio>
#include <mpi.h>
#include <unordered_map>

void RandomForest::mpi_fit(const std::vector<std::vector<double>>& X,
                           const std::vector<uint32_t>& y)
{
    omp_fit(X, y);
}

std::vector<uint32_t> RandomForest::mpi_predict(
    const std::vector<std::vector<double>>& X)
{
    std::vector<uint32_t> y = omp_predict(X);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    if (rank == 0)
    {
        uint32_t* buffer = new uint32_t[X.size()];
        std::vector<std::vector<uint32_t>> predictions;
        predictions.reserve(n_proc);
        predictions.push_back(y);
        for (int i = 1; i < n_proc; i++)
        {
            MPI_Recv(buffer, X.size(), MPI_UINT32_T, i, 1, MPI_COMM_WORLD,
                     nullptr);

            predictions.emplace_back(buffer, buffer + X.size());
        }
        delete[] buffer;

        // count votes and compute majority
        std::vector<uint32_t> prediction(X.size());
#pragma omp parallel for num_threads(m_Threads)
        for (size_t i = 0; i < X.size(); i++)
        {
            std::unordered_map<uint32_t, size_t> counter;
            for (int j = 0; j < n_proc; j++)
            {
                const std::vector<uint32_t>& pred = predictions[j];
                counter[pred[i]]++;
            }

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
            prediction[i] = value;
        }

        return prediction;
    }
    else
    {
        MPI_Send(y.data(), y.size(), MPI_UINT32_T, 0, 1, MPI_COMM_WORLD);
        return {};
    }
}
