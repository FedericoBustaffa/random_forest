#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <vector>

#include "tensor_view.hpp"

class DecisionTree
{
public:
    DecisionTree();

    void fit(const TensorView& X, const TensorView& y);

    // TensorView predict(const TensorView& X);

    ~DecisionTree();

private:
    double entropy(const TensorView& y);

    double information_gain(const TensorView& feature, const TensorView& y,
                            double threshold);

    void split(size_t feature, double threshold);

private:
    struct Node
    {
        Node(size_t feature, double threshold)
            : m_Feature(feature), m_Threshold(threshold), m_Left(nullptr),
              m_Right(nullptr)
        {
        }

        size_t m_Feature;
        double m_Threshold;
        Node* m_Left;
        Node* m_Right;
    };

private:
    std::vector<Node> nodes;
};

#endif
