#include "random_forest.hpp"

#include <mpi.h>
#include <omp.h>

#include "utils.hpp"

void RandomForest::mpi_fit(const std::vector<std::vector<double>>& X,
                           const std::vector<uint32_t>& y)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto T = transpose(X);
#pragma omp parallel for schedule(dynamic) num_threads(m_Threads)
    for (size_t i = 0; i < m_Trees.size(); i++)
    {

        printf("Rank %d: I am thread %d of %d\n", rank, omp_get_thread_num(),
               omp_get_num_threads());

        uint32_t seed = rank * m_Trees.size() + i;
        std::vector<size_t> indices = bootstrap(T[0].size(), seed);
        m_Trees[i].fit(T, y, indices);
    }
}

std::vector<uint32_t> RandomForest::mpi_predict(
    const std::vector<std::vector<double>>& X)
{
    // predict the same batch in parallel
    std::vector<uint32_t> y(X.size() * m_Trees.size());
#pragma omp parallel for num_threads(m_Threads)
    for (size_t i = 0; i < m_Trees.size(); i++)
    {
        std::vector<uint32_t> single = m_Trees[i].predict(X);
        for (size_t j = 0; j < X.size(); j++)
            y[j * m_Trees.size() + i] = single[j];
    }

    // count votes
    std::vector<uint64_t> counters(X.size() * m_Labels);
#pragma omp parallel for num_threads(m_Threads)
    for (size_t i = 0; i < X.size(); i++)
    {
        size_t row = i * m_Labels;
        for (size_t j = 0; j < m_Trees.size(); j++)
        {
            uint32_t label = y[i * m_Trees.size() + j];
            counters[row + label]++;
        }
    }

    std::vector<uint64_t> buffer(X.size() * m_Labels);
    MPI_Reduce(counters.data(), buffer.data(), counters.size(), MPI_UINT64_T,
               MPI_SUM, 0, MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::vector<uint32_t> prediction(X.size());
#pragma omp parallel for num_threads(m_Threads)
        for (size_t i = 0; i < X.size(); i++)
        {
            size_t best = 0;
            size_t row = i * m_Labels;
            for (size_t j = 1; j < m_Labels; j++)
            {
                if (buffer[row + j] > buffer[row + best])
                    best = j;
            }

            prediction[i] = best;
        }

        return prediction;
    }

    return {};
}
