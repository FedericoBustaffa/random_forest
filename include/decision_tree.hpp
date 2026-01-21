#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "view.hpp"

class DecisionTree
{
public:
    DecisionTree(size_t max_depth = 0, bool bootstrap = false,
                 int64_t random_state = -1);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<uint32_t>& y);

    std::vector<uint32_t> predict(const std::vector<std::vector<double>>& X);

    size_t size() const { return m_Tree.size(); }

    size_t depth() const { return compute_depth(0); }

    ~DecisionTree();

private:
    struct Node
    {
        Node(uint32_t feature, double threshold)
            : feature(feature), threshold(threshold), label(-1)
        {
        }

        Node(int label) : feature(0), threshold(0.0), label(label) {}

        uint32_t feature;
        double threshold;
        int label;

        int64_t left = -1;
        int64_t right = -1;
    };

private: // tree private methods
    int64_t grow(const std::vector<View<double>>& X, const View<uint32_t>& y,
                 size_t n_labels, size_t depth);

    uint32_t predict_one(const std::vector<double>& x, int64_t i);

    size_t compute_depth(int64_t i) const;

private: // tree data members
    std::vector<Node> m_Tree;
    size_t m_MaxDepth;
    bool m_Bootstrap;
    int64_t m_RandomState;
};

#endif
