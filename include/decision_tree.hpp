#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

class DecisionTree
{
public:
    DecisionTree(size_t max_depth = 0);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<uint32_t>& y,
             const std::vector<size_t>& indices);

    std::vector<uint32_t> predict(const std::vector<std::vector<double>>& X);

    size_t depth() const;

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

        // indices
        int64_t left = -1;
        int64_t right = -1;
    };

private: // tree private methods
    void grow(const std::vector<std::vector<double>>& X,
              const std::vector<uint32_t>& y, std::vector<size_t> indices,
              size_t depth);

    uint32_t predict_one(int64_t idx, const std::vector<double>& x);

    size_t compute_depth(int64_t idx) const;

private: // tree data members
    std::vector<Node> m_Tree;
    size_t m_MaxDepth;
};

#endif
