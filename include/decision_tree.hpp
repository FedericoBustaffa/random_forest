#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "datasplit.hpp"

class DecisionTree
{
public:
    DecisionTree(size_t max_depth = 0, bool bootstrap = false,
                 int64_t random_state = -1);

    inline size_t max_depth() const { return m_MaxDepth; }

    inline size_t size() const { return m_Tree.size(); }

    size_t depth() const { return compute_depth(0); }

    void fit(const DataSplit& data);

    std::vector<uint8_t> predict(const std::vector<std::vector<float>>& X);

    ~DecisionTree();

private:
    struct Node
    {
        Node(size_t feature_idx, float threshold)
            : feature_idx(feature_idx), threshold(threshold)
        {
        }

        Node(int label) : label(label) {}

        inline bool is_leaf() const { return left == -1 && right == -1; }

        size_t feature_idx = 0;
        float threshold = 0.0;
        uint8_t label = 0;

        int64_t left = -1;
        int64_t right = -1;
    };

private: // tree private methods
    int64_t grow(const std::vector<std::vector<float>>& X,
                 const std::vector<uint8_t>& y,
                 const std::vector<FeatureType>& types,
                 std::vector<size_t>& indices, size_t n_labels, size_t depth);

    size_t compute_depth(int64_t i) const;

private: // tree data members
    size_t m_MaxDepth;
    bool m_Bootstrap;
    int64_t m_RandomState;

    std::vector<Node> m_Tree;
};

#endif
