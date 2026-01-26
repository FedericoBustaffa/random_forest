#include "random_forest.hpp"

#include <mpi.h>

#include "utils.hpp"

RandomForest::RandomForest(size_t estimators, size_t max_depth, Backend backend,
                           size_t threads)
    : m_Backend(backend), m_Threads(threads)
{
    if (m_Backend != Backend::MPI)
        m_Nodes = 1;
    else
        MPI_Comm_size(MPI_COMM_WORLD, (int*)&m_Nodes);

    size_t ntrees = estimators / m_Nodes;
    m_Trees.reserve(ntrees);
    for (size_t i = 0; i < ntrees; i++)
        m_Trees.emplace_back(max_depth, true, ntrees + i);
}

void RandomForest::fit(const DataSplit& data)
{
    // save the possible number of labels/classes
    m_Labels = count_labels(data.y_train);

    // fit based on the chosen backend
    switch (m_Backend)
    {
    case Backend::Sequential:
        seq_fit(data);
        break;

    case Backend::OpenMP:
        omp_fit(data);
        break;

    case Backend::FastFlow:
        ff_fit(data);
        break;

    case Backend::MPI:
        mpi_fit(data);
        break;

    default:
        break;
    }
}

std::vector<uint8_t> RandomForest::predict(
    const std::vector<std::vector<float>>& X)
{
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
    for (size_t i = 0; i < m_Trees.size(); i++)
        out[i] = m_Trees[i].depth();

    return out;
}

RandomForest::~RandomForest() {}
