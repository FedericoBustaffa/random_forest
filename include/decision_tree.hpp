#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "view.hpp"

class DecisionTree
{
public:
    DecisionTree();

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<unsigned int>& y);

    std::vector<unsigned int> predict(
        const std::vector<std::vector<double>>& X);

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

private:
    double entropy(const View<unsigned int>& y);

    double entropy(const std::unordered_map<size_t, size_t>& counters,
                   size_t size);

    double informationGain(const View<double>& feature,
                           const View<unsigned int>& y, double threshold);

    double informationGain(
        const std::array<std::unordered_map<size_t, size_t>, 2> counters);

    Node* grow(Node* root, const std::vector<View<double>>& X,
               const View<unsigned int>& y);

    unsigned int visit(Node* node, const std::vector<double>& x);

    void deallocate(Node* node);

private:
    Node* m_Root;
};

#endif
