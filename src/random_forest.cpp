#include "random_forest.hpp"

#include "utils.hpp"

RandomForest::RandomForest(size_t estimators, size_t max_depth, Backend backend,
                           size_t threads, size_t nodes)
    : m_Backend(backend), m_Threads(threads), m_Nodes(nodes)
{
    size_t ntrees = estimators / nodes;
    m_Trees.reserve(ntrees);
    for (size_t i = 0; i < ntrees; i++)
        m_Trees.emplace_back(max_depth, true, ntrees + i);
}

void RandomForest::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<uint32_t>& y)
{
    // save the possible number of labels/classes
    m_Labels = count_labels(y);

    // fit based on the chosen backend
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

std::vector<uint32_t> RandomForest::predict(
    const std::vector<std::vector<double>>& X)
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
