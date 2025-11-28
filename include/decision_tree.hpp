#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "view.hpp"

class DecisionTree
{
public:
    DecisionTree(size_t max_depth = 0);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<uint32_t>& y);

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

        Node* left = nullptr;
        Node* right = nullptr;
    };

private: // tree private methods
    Node* grow(Node* root, const std::vector<View<double>>& X,
               const View<uint32_t>& y, size_t depth);

    uint32_t predict_one(Node* node, const std::vector<double>& x);

    size_t compute_depth(Node* node) const;

    void deallocate(Node* node);

private: // tree data members
    Node* m_Root = nullptr;
    size_t m_MaxDepth;
};

#endif
