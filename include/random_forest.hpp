#ifndef RANDOM_FOREST_HPP
#define RANDOM_FOREST_HPP

#include <vector>

#include "backend.hpp"
#include "decision_tree.hpp"

class RandomForest
{
public:
    RandomForest(size_t estimators, size_t max_depth = 0,
                 Backend backend = Backend::Sequential, size_t threads = 1);

    inline size_t estimators() const { return m_Trees.size(); }

    inline size_t max_depth() const { return m_Trees[0].max_depth(); }

    inline Backend backend() const { return m_Backend; }

    inline size_t thrads() const { return m_Threads; }

    inline size_t nodes() const { return m_Nodes; }

    void fit(const std::vector<std::vector<float>>& X,
             const std::vector<uint8_t>& y);

    std::vector<uint8_t> predict(const std::vector<std::vector<float>>& X);

    std::vector<size_t> depths() const;

    ~RandomForest();

private:
    void seq_fit(const std::vector<std::vector<float>>& X,
                 const std::vector<uint8_t>& y);

    void omp_fit(const std::vector<std::vector<float>>& X,
                 const std::vector<uint8_t>& y);

    void ff_fit(const std::vector<std::vector<float>>& X,
                const std::vector<uint8_t>& y);

    void mpi_fit(const std::vector<std::vector<float>>& X,
                 const std::vector<uint8_t>& y);

    std::vector<uint8_t> seq_predict(const std::vector<std::vector<float>>& X);

    std::vector<uint8_t> omp_predict(const std::vector<std::vector<float>>& X);

    std::vector<uint8_t> ff_predict(const std::vector<std::vector<float>>& X);

    std::vector<uint8_t> mpi_predict(const std::vector<std::vector<float>>& X);

private:
    std::vector<DecisionTree> m_Trees;
    uint8_t m_Labels = 0;

    Backend m_Backend;
    size_t m_Threads;
    size_t m_Nodes;
};

#endif
