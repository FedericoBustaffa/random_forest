#include "random_forest.hpp"

#include <mpi.h>

#include "utils.hpp"

RandomForest::RandomForest(size_t estimators, size_t max_depth, Backend backend,
                           size_t threads)
    : m_Backend(backend), m_Threads(threads)
{
    m_Nodes = 1;

    // Determine number of MPI nodes if using MPI backend
    if (m_Backend == Backend::MPI)
    {
        int nodes;
        MPI_Comm_size(MPI_COMM_WORLD, &nodes);
        m_Nodes = nodes;
    }

    // Initialize trees for this process/node
    size_t ntrees = estimators / m_Nodes;
    m_Trees.reserve(ntrees);
    for (size_t i = 0; i < ntrees; i++)
        m_Trees.emplace_back(max_depth, true, ntrees + i);
}

void RandomForest::fit(const std::vector<std::vector<float>>& X,
                       const std::vector<uint8_t>& y)
{
    // Count number of unique labels/classes
    m_Labels = count_labels(y);

    // Fit trees based on chosen backend
    switch (m_Backend)
    {
    case Backend::Sequential:
        seq_fit(X, y);
        break;

    case Backend::OpenMP:
        omp_fit(X, y);
        break;

    case Backend::FastFlow:
        ff_fit(X, y);
        break;

    case Backend::MPI:
        mpi_fit(X, y);
        break;

    default:
        break;
    }
}

std::vector<uint8_t> RandomForest::predict(
    const std::vector<std::vector<float>>& X)
{
    // Dispatch prediction based on backend
    switch (m_Backend)
    {
    case Backend::Sequential:
        return seq_predict(X);

    case Backend::OpenMP:
        return omp_predict(X);

    case Backend::FastFlow:
        return ff_predict(X);

    case Backend::MPI:
        return mpi_predict(X);

    default:
        return {};
    }
}

std::vector<size_t> RandomForest::depths() const
{
    std::vector<size_t> out(m_Trees.size());

    // Compute depth for each tree
    for (size_t i = 0; i < m_Trees.size(); i++)
        out[i] = m_Trees[i].depth();

    return out;
}

RandomForest::~RandomForest() {}
