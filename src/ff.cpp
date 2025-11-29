#include "random_forest.hpp"

#include <ff/ff.hpp>

#include "utils.hpp"

using namespace ff;

struct Task
{
    const std::vector<std::vector<double>>& X;
    const std::vector<uint32_t>& y;
    DecisionTree& tree;
};

class Source : public ff_node_t<Task>
{
public:
    Source(const std::vector<std::vector<double>>& X,
           const std::vector<uint32_t>& y, std::vector<DecisionTree>& trees)
        : X(X), y(y), trees(trees)
    {
    }

    Task* svc(Task* task) override
    {
        for (size_t i = 0; i < trees.size(); i++)
            ff_send_out(new Task(X, y, trees[i]));

        return EOS;
    }

private:
    const std::vector<std::vector<double>>& X;
    const std::vector<uint32_t>& y;
    std::vector<DecisionTree>& trees;
};

class Worker : public ff_node_t<Task>
{
public:
    Task* svc(Task* task) override
    {
        std::vector<size_t> indices = bootstrap(task->X[0].size());
        task->tree.fit(task->X, task->y, indices);
        delete task;

        return GO_ON;
    }
};

void RandomForest::ff_fit(const std::vector<std::vector<double>>& X,
                          const std::vector<uint32_t> y)
{
    auto T = transpose(X);

    Source source(T, y, m_Trees);

    std::vector<std::unique_ptr<ff_node>> workers;
    for (size_t i = 0; i < m_Threads; i++)
        workers.push_back(std::make_unique<Worker>());

    ff_Farm<Task> farm(std::move(workers), source);
    farm.remove_collector();
    farm.run_and_wait_end();
}

std::vector<uint32_t> RandomForest::ff_predict(
    const std::vector<std::vector<double>>& X)
{
    // predict the same batch in parallel
    std::vector<std::vector<uint32_t>> y(m_Trees.size());
#pragma omp parallel for
    for (size_t i = 0; i < m_Trees.size(); i++)
        y[i] = m_Trees[i].predict(X);

    // count votes
    std::vector<std::unordered_map<uint32_t, size_t>> counters(y[0].size());
#pragma omp parallel for
    for (size_t i = 0; i < counters.size(); i++)
    {
        for (size_t j = 0; j < y.size(); j++)
        {
            const std::vector<uint32_t>& pred = y[j];
            counters[i][pred[i]]++;
        }
    }

    // compute majority
    std::vector<uint32_t> prediction(counters.size());
#pragma omp parallel for
    for (size_t i = 0; i < counters.size(); i++)
    {
        uint32_t value = 0;
        size_t counter = 0;
        for (const auto& kv : counters[i])
        {
            if (kv.second > counter)
            {
                counter = kv.second;
                value = kv.first;
            }
        }
        prediction[i] = value;
    }

    return prediction;
}
