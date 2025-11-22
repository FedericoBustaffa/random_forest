#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include "tensor_view.hpp"

class DecisionTree
{
public:
    DecisionTree();

    void fit(const TensorView& X, const TensorView& y);

    // TensorView predict(const TensorView& X);

    ~DecisionTree();

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
    double entropy(const TensorView& y);

    double informationGain(const TensorView& feature, const TensorView& y,
                           double threshold);

    Node* grow(Node* node, const TensorView& X, const TensorView& y);

    void deallocate(Node* node);

private:
    Node* m_Root;
};

#endif
