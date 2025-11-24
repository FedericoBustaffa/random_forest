#include "vector_view.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

VectorView::VectorView(const double* data, size_t size, size_t stride,
                       const std::vector<size_t>& indices)
    : m_View(data), m_Size(size), m_Stride(stride), m_Indices(indices)
{
}

VectorView::VectorView(const double* data, size_t size)
    : m_View(data), m_Size(size), m_Stride(1), m_Indices(size)
{
    for (size_t i = 0; i < m_Size; i++)
        m_Indices[i] = i;
}

VectorView::VectorView(const VectorView& other)
    : m_View(other.m_View), m_Size(other.m_Size), m_Stride(other.m_Stride),
      m_Indices(other.m_Indices)
{
}

VectorView::VectorView(VectorView&& other) noexcept
    : m_View(other.m_View), m_Size(other.m_Size), m_Stride(other.m_Stride),
      m_Indices(std::move(other.m_Indices))
{
    other.m_View = nullptr;
    other.m_Size = 0;
}

VectorView& VectorView::operator=(const VectorView& other)
{
    if (this != &other)
    {
        m_View = other.m_View;
        m_Size = other.m_Size;
        m_Stride = other.m_Stride;
        m_Indices = other.m_Indices;
    }

    return *this;
}

VectorView& VectorView::operator=(VectorView&& other) noexcept
{
    if (this != &other)
    {
        m_View = other.m_View;
        m_Size = other.m_Size;
        m_Stride = other.m_Stride;
        m_Indices = std::move(other.m_Indices);

        other.m_View = nullptr;
        other.m_Size = 0;
        other.m_Stride = 0;
    }

    return *this;
}

VectorView VectorView::operator[](const std::vector<size_t>& indices) const
{
    std::vector<size_t> new_indices(indices.size());
    for (size_t i = 0; i < indices.size(); i++)
        new_indices[i] = m_Indices[indices[i]];

    return VectorView(m_View, new_indices.size(), m_Stride, new_indices);
}

Mask VectorView::operator<=(double value) const
{
    std::vector<bool> mask(m_Size);
    for (size_t i = 0; i < m_Size; i++)
        mask[i] = (m_View[m_Indices[i] * m_Stride] <= value);

    return Mask(mask);
}

VectorView VectorView::operator[](const Mask& mask) const
{
    std::vector<size_t> indices;
    for (size_t i = 0; i < m_Size; i++)
        if (mask[i])
            indices.push_back(m_Indices[i]);

    return VectorView(m_View, indices.size(), m_Stride, indices);
}
