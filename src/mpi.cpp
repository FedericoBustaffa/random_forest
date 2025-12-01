#include "random_forest.hpp"

#include <mpi.h>

void RandomForest::mpi_fit(const std::vector<std::vector<double>>& X,
                           const std::vector<uint32_t> y)
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
    }
    else
    {
    }

    std::vector<uint32_t> prediction;

    return prediction;
}
