#ifndef RANDOM_FOREST_HPP
#define RANDOM_FOREST_HPP

#include <cstdint>
#include <vector>

#include "decision_tree.hpp"

enum class Backend
{
    Sequential,
    OpenMP,
    FastFlow,
    MPI,
    Invalid
};

class RandomForest
{
public:
    RandomForest(size_t estimators, size_t max_depth = 0,
                 Backend backend = Backend::Sequential, size_t n_threads = 1,
                 size_t nodes = 1);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<uint32_t> y);

    std::vector<uint32_t> predict(const std::vector<std::vector<double>>& X);

    std::vector<size_t> depths() const;

    ~RandomForest();

private:
    void seq_fit(const std::vector<std::vector<double>>& X,
                 const std::vector<uint32_t> y);

    void omp_fit(const std::vector<std::vector<double>>& X,
                 const std::vector<uint32_t> y);

    void ff_fit(const std::vector<std::vector<double>>& X,
                const std::vector<uint32_t> y);

    std::vector<uint32_t> seq_predict(
        const std::vector<std::vector<double>>& X);

    std::vector<uint32_t> omp_predict(
        const std::vector<std::vector<double>>& X);

    std::vector<uint32_t> ff_predict(const std::vector<std::vector<double>>& X);

private:
    std::vector<DecisionTree> m_Trees;
    Backend m_Backend;
    size_t m_Threads;
    size_t m_Nodes;
};

#endif
