#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

class DecisionTree
{
public:
    DecisionTree(size_t max_depth = 0, bool bootstrap = false,
                 int64_t random_state = -1);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<uint32_t>& y);

    std::vector<uint32_t> predict(const std::vector<std::vector<double>>& X);

    size_t size() const { return compute_size(m_Root); }

    size_t depth() const { return compute_depth(m_Root); }

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
    Node* grow(Node* root, const std::vector<std::vector<double>>& X,
               const std::vector<uint32_t>& y,
               const std::vector<size_t>& indices, size_t depth);

    uint32_t predict_one(Node* node, const std::vector<double>& x);

    size_t compute_size(Node* node) const;

    size_t compute_depth(Node* node) const;

    void deallocate(Node* node);

private: // tree data members
    Node* m_Root = nullptr;
    size_t m_MaxDepth;
    bool m_Bootstrap;
    int64_t m_RandomState;
};

#endif
