#include "random_forest.hpp"

#include <ff/ff.hpp>

#include <cstddef>
#include <cstdint>

#include "counter.hpp"

using namespace ff;

// Source node for farm: generates job IDs
class Source : public ff_node_t<size_t>
{
public:
    Source(size_t n_jobs) : n_jobs(n_jobs) {}

    size_t* svc(size_t* in) override
    {
        for (size_t i = 0; i < n_jobs; i++)
            ff_send_out(new size_t(i));

        return EOS;
    }

private:
    size_t n_jobs;
};

// Worker node: fits a single tree
class Fitter : public ff_node_t<size_t>
{
public:
    Fitter(const std::vector<std::vector<float>>& X,
           const std::vector<uint8_t>& y, std::vector<DecisionTree>& trees)
        : X(X), y(y), trees(trees)
    {
    }

    size_t* svc(size_t* i) override
    {
        trees[*i].fit(X, y);
        delete i;

        return GO_ON;
    }

private:
    const std::vector<std::vector<float>>& X;
    const std::vector<uint8_t>& y;
    std::vector<DecisionTree>& trees;
};

// Fit all trees using FastFlow farm
void RandomForest::ff_fit(const std::vector<std::vector<float>>& X,
                          const std::vector<uint8_t>& y)
{
    Source source(m_Trees.size());

    // create worker nodes
    std::vector<std::unique_ptr<ff_node>> workers;
    for (size_t i = 0; i < m_Threads; i++)
        workers.push_back(std::make_unique<Fitter>(X, y, m_Trees));

    ff_Farm<size_t> farm(std::move(workers), source);
    farm.remove_collector();
    farm.run_and_wait_end();
}

// Worker node: predicts with a single tree
class Predicter : public ff_node_t<size_t>
{
public:
    Predicter(const std::vector<std::vector<float>>& X,
              std::vector<std::vector<uint8_t>>& y,
              std::vector<DecisionTree>& trees)
        : X(X), y(y), trees(trees)
    {
    }

    size_t* svc(size_t* i) override
    {
        y[*i] = trees[*i].predict(X);
        delete i;

        return GO_ON;
    }

private:
    const std::vector<std::vector<float>>& X;
    std::vector<std::vector<uint8_t>>& y;
    std::vector<DecisionTree>& trees;
};

// Worker node: counts votes and computes majority
class VoteCounter : public ff_node_t<size_t>
{
public:
    VoteCounter(const std::vector<std::vector<uint8_t>>& y,
                std::vector<Counter>& votes, std::vector<uint8_t>& prediction)
        : y(y), votes(votes), prediction(prediction)
    {
    }

    size_t* svc(size_t* i) override
    {
        // aggregate votes for this tree
        for (size_t j = 0; j < y.size(); j++)
        {
            const std::vector<uint8_t>& pred = y[j];
            votes[*i][pred[*i]]++;
        }

        // compute majority class
        uint8_t value = 0;
        size_t counter = 0;
        for (size_t j = 0; j < votes[*i].size(); ++j)
        {
            if (votes[*i][j] > counter)
            {
                counter = votes[*i][j];
                value = j;
            }
        }
        prediction[*i] = value;

        delete i;

        return GO_ON;
    }

private:
    const std::vector<std::vector<uint8_t>>& y;
    std::vector<Counter>& votes;
    std::vector<uint8_t>& prediction;
};

// Predict all samples using FastFlow farm
std::vector<uint8_t> RandomForest::ff_predict(
    const std::vector<std::vector<float>>& X)
{
    // predict with all trees in parallel
    Source tree_source(m_Trees.size());
    std::vector<std::vector<uint8_t>> y(m_Trees.size());

    std::vector<std::unique_ptr<ff_node>> predicters;
    for (size_t i = 0; i < m_Threads; i++)
        predicters.push_back(std::make_unique<Predicter>(X, y, m_Trees));
    ff_Farm<size_t> predict_farm(std::move(predicters), tree_source);

    predict_farm.remove_collector();
    predict_farm.run_and_wait_end();

    // count votes and compute majority
    std::vector<Counter> votes(X.size(), m_Labels);
    std::vector<uint8_t> prediction(votes.size());

    Source votes_source(votes.size());
    std::vector<std::unique_ptr<ff_node>> counters;
    for (size_t i = 0; i < m_Threads; i++)
        counters.push_back(std::make_unique<VoteCounter>(y, votes, prediction));
    ff_Farm<size_t> counter_farm(std::move(counters), votes_source);

    counter_farm.remove_collector();
    counter_farm.run_and_wait_end();

    return prediction;
}
