#ifndef VECTOR_VIEW_HPP
#define VECTOR_VIEW_HPP

#include <cassert>
#include <cstddef>
#include <vector>

#include "mask.hpp"

class VectorView
{
public:
    VectorView(const double* data, size_t size, size_t stride,
               const std::vector<size_t>& indices)
        : m_View(data), m_Size(size), m_Stride(stride), m_Indices(indices)
    {
    }

    VectorView(const double* data, size_t size)
        : m_View(data), m_Size(size), m_Stride(1), m_Indices(size)
    {
        for (size_t i = 0; i < m_Size; i++)
            m_Indices[i] = i;
    }

    VectorView(const VectorView& other)
        : m_View(other.m_View), m_Size(other.m_Size), m_Stride(other.m_Stride),
          m_Indices(other.m_Indices)
    {
    }

    VectorView(VectorView&& other) noexcept
        : m_View(other.m_View), m_Size(other.m_Size), m_Stride(other.m_Stride),
          m_Indices(std::move(other.m_Indices))
    {
        other.m_View = nullptr;
        other.m_Size = 0;
    }

    VectorView& operator=(const VectorView& other)
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

    VectorView& operator=(VectorView&& other) noexcept
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

    virtual inline size_t size() const { return m_Size; }

    virtual inline double operator[](size_t idx) const
    {
        return m_View[m_Indices[idx] * m_Stride];
    }

    VectorView operator[](const std::vector<size_t>& indices) const
    {
        std::vector<size_t> new_indices(indices.size());
        for (size_t i = 0; i < indices.size(); i++)
            new_indices[i] = m_Indices[indices[i]];

        return VectorView(m_View, new_indices.size(), m_Stride, new_indices);
    }

    Mask operator<=(double value) const
    {
        std::vector<bool> mask(m_Size);
        for (size_t i = 0; i < m_Size; i++)
            mask[i] = (m_View[m_Indices[i] * m_Stride] <= value);

        return Mask(mask);
    }

    VectorView operator[](const Mask& mask) const
    {
        std::vector<size_t> indices;
        for (size_t i = 0; i < m_Size; i++)
            if (mask[i])
                indices.push_back(m_Indices[i]);

        return VectorView(m_View, indices.size(), m_Stride, indices);
    }

protected:
    const double* m_View = nullptr;
    size_t m_Size = 0;
    size_t m_Stride = 0;
    std::vector<size_t> m_Indices;
};

#endif
