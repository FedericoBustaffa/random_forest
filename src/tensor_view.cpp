#include "tensor_view.hpp"

#include <cassert>
#include <stdexcept>

TensorView::TensorView(const double* data, const std::vector<size_t>& shape,
                       const std::vector<size_t>& indices,
                       const std::vector<size_t>& strides)
    : m_Shape(shape), m_Indices(indices), m_Strides(strides), m_View(data)
{
    // compute the total size of the tensor
    m_Size = 1;
    for (const auto& d : shape)
        m_Size *= d;
}

TensorView::TensorView(const double* data, const std::vector<size_t>& shape)
    : m_Shape(shape), m_Strides(shape.size(), 0), m_View(data)
{
    // compute the total size of the tensor
    m_Size = 1;
    for (const auto& d : shape)
        m_Size *= d;

    // if not specified the indexing is redundant
    m_Indices.reserve(m_Size);
    for (size_t i = 0; i < m_Size; i++)
        m_Indices.push_back(i);

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
}

TensorView::TensorView(const TensorView& other)
    : m_Shape(other.m_Shape), m_Indices(other.m_Indices),
      m_Strides(other.m_Strides), m_Size(other.m_Size), m_View(other.m_View)
{
}

TensorView::TensorView(TensorView&& other) noexcept
    : m_Shape(std::move(other.m_Shape)), m_Indices(std::move(other.m_Indices)),
      m_Strides(std::move(other.m_Strides)), m_Size(other.m_Size),
      m_View(other.m_View)
{
    other.m_Size = 0;
    other.m_View = nullptr;
}

TensorView& TensorView::operator=(const TensorView& other)
{
    m_Shape = other.m_Shape;
    m_Indices = other.m_Indices;
    m_Strides = other.m_Strides;
    m_Size = other.m_Size;
    m_View = other.m_View;

    return *this;
}

TensorView& TensorView::operator=(TensorView&& other) noexcept
{
    m_Shape = std::move(other.m_Shape);
    m_Indices = std::move(other.m_Indices);
    m_Strides = std::move(other.m_Strides);
    m_Size = other.m_Size;
    m_View = other.m_View;

    return *this;
}

TensorView TensorView::operator[](size_t idx) const
{
    if (m_Shape.empty())
        throw std::runtime_error("scalar is not subscriptable");

    if (idx >= m_Shape[0])
        throw std::out_of_range("index out of bounds");

    std::vector<size_t> new_shape(m_Shape.begin() + 1, m_Shape.end());
    std::vector<size_t> new_strides(m_Strides.begin() + 1, m_Strides.end());

    std::vector<size_t> new_indices;
    if (!new_shape.empty())
    {
        for (size_t i = 0; i < new_shape[0]; i++)
            new_indices.push_back(i);
    }

    const double* new_view = m_View + m_Indices[idx] * m_Strides[0];

    return TensorView(new_view, new_shape, new_indices, new_strides);
}

TensorView TensorView::operator[](const std::vector<bool>& mask) const
{
    assert(mask.size() == m_Shape[0]);

    std::vector<size_t> idx;
    idx.reserve(m_Shape[0]);

    for (size_t i = 0; i < mask.size(); i++)
        if (mask[i])
            idx.push_back(i);

    return (*this)[idx];
}

TensorView TensorView::operator[](const std::vector<size_t>& indices) const
{
    return TensorView(m_View, m_Shape, indices, m_Strides);
}

TensorView TensorView::operator()(size_t idx, size_t axis) const
{
    if (m_Shape.empty())
        throw std::runtime_error("scalar is not supscriptable");

    std::vector<size_t> new_shape = m_Shape;
    new_shape.erase(new_shape.begin() + axis);

    std::vector<size_t> new_indices;
    for (size_t i = 0; i < m_Shape[axis]; i++)
        new_indices.push_back(m_Indices[i * m_Strides[axis] + idx]);

    std::vector<size_t> new_strides = m_Strides;
    new_strides.erase(new_strides.begin() + axis);

    const double* new_view = m_View + m_Indices[idx] * m_Strides[axis];

    return TensorView(new_view, new_shape, new_indices, new_strides);
}

TensorView::~TensorView() {}
