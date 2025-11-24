#ifndef VECTOR_VIEW_HPP
#define VECTOR_VIEW_HPP

#include <cassert>
#include <cstddef>
#include <vector>

#include "mask.hpp"

class Vector;

class VectorView
{
public:
    VectorView(const double* data, size_t size, size_t stride,
               const std::vector<size_t>& indices);

    VectorView(const double* data, size_t size);

    VectorView(const VectorView& other);

    VectorView(VectorView&& other) noexcept;

    VectorView& operator=(const VectorView& other);

    VectorView& operator=(VectorView&& other) noexcept;

    virtual inline size_t size() const { return m_Size; }

    Vector copy() const;

    virtual inline double operator[](size_t idx) const
    {
        return m_View[m_Indices[idx] * m_Stride];
    }

    VectorView operator[](const std::vector<size_t>& indices) const;

    Mask operator<=(double value) const;

    VectorView operator[](const Mask& mask) const;

protected:
    const double* m_View = nullptr;
    size_t m_Size = 0;
    size_t m_Stride = 0;
    std::vector<size_t> m_Indices;
};

#endif
