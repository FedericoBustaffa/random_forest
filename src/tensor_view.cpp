#include "tensor_view.hpp"

#include <stdexcept>

TensorView::TensorView(const double* data, const std::vector<size_t>& shape)
    : m_Shape(shape), m_Strides(m_Shape.size(), 0)
{
    // compute the total size of the tensor
    m_Size = 1;
    for (const auto& d : shape)
        m_Size *= d;

    // if not a scalar compute the strides
    if (!m_Shape.empty())
    {
        // 1D vector
        m_Strides.back() = 1;
        if (m_Shape.size() > 1)
        {
            // N-dimensional tensor
            for (size_t i = m_Shape.size() - 1; i-- > 0;)
                m_Strides[i] = m_Strides[i + 1] * m_Shape[i + 1];
        }
    }

    m_View = data;
}

TensorView::TensorView(const TensorView& other)
    : m_Shape(other.m_Shape), m_Size(other.m_Size), m_Strides(other.m_Strides),
      m_View(other.m_View)
{
}

void TensorView::operator=(const TensorView& other)
{
    m_Shape = other.m_Shape;
    m_Size = other.m_Size;
    m_Strides = other.m_Strides;
    m_View = other.m_View;
}

TensorView TensorView::operator[](size_t i) const
{
    // try to use [] on a scalar
    if (m_Shape.empty())
        throw std::runtime_error("scalar is not supscriptable");

    std::vector<size_t> shape(m_Shape.begin() + 1, m_Shape.end());
    const double* data = m_View + i * m_Strides[0];

    return TensorView(data, shape);
}

TensorView::~TensorView() {}
