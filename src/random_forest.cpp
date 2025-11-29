#include "random_forest.hpp"

#include <cstdio>
#include <stdexcept>

RandomForest::RandomForest(size_t estimators, size_t max_depth, Backend backend,
                           size_t n_workers)
    : m_Trees(estimators, max_depth), m_Backend(backend), m_Threads(n_workers)
{
}

void RandomForest::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<uint32_t> y)
{
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
        throw std::runtime_error("MPI backend not supported yet");

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
        throw std::runtime_error("MPI backend not supported yet");

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
