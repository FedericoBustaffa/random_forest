#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include "matrix_view.hpp"
#include "vector.hpp"

class DecisionTree
{
public:
    DecisionTree();

    void fit(const MatrixView& X, const VectorView& y);

    Vector predict(const MatrixView& X);

    ~DecisionTree();

private:
    struct Node
    {
        Node(size_t feature, double threshold)
            : m_Feature(feature), m_Threshold(threshold), m_Class(-1),
              m_Left(nullptr), m_Right(nullptr)
        {
        }

        size_t m_Feature;
        double m_Threshold;
        double m_Class;
        Node* m_Left;
        Node* m_Right;
    };

private:
    double entropy(const VectorView& y);

    double informationGain(const VectorView& feature, const VectorView& y,
                           double threshold);

    Node* grow(Node* node, const MatrixView& X, const VectorView& y);

    double visit(Node* node, VectorView x);

    void deallocate(Node* node);

private:
    Node* m_Root;
};

#endif
