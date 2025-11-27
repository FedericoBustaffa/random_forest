#ifndef VIEW_HPP
#define VIEW_HPP

#include <cassert>
#include <cstddef>

#include "mask.hpp"

template <typename T>
class View
{
public:
    View() = default;

    View(const T* data, size_t size) : m_Data(data), m_Indices(size)
    {
        for (size_t i = 0; i < size; i++)
            m_Indices[i] = i;
    }

    View(const T* data, size_t size, std::vector<size_t> indices)
        : m_Data(data), m_Indices(indices)
    {
    }

    inline size_t size() const { return m_Indices.size(); }

    T operator[](size_t idx) const
    {
        assert(idx < m_Indices.size());
        return m_Data[m_Indices[idx]];
    }

    View<T> operator[](const std::vector<size_t>& indices) const
    {
        std::vector<size_t> new_indices(indices.size());
        for (size_t i = 0; i < indices.size(); i++)
            new_indices[i] = m_Indices[indices[i]];

        return View<T>(m_Data, new_indices.size(), new_indices);
    }

    Mask operator<(T value) const
    {
        std::vector<bool> mask(m_Indices.size());
        for (size_t i = 0; i < m_Indices.size(); i++)
            mask[i] = m_Data[m_Indices[i]] < value;

        return Mask(mask);
    }

    View<T> operator[](const Mask& mask) const
    {
        std::vector<size_t> indices;
        for (size_t i = 0; i < m_Indices.size(); i++)
        {
            if (mask[i])
                indices.push_back(m_Indices[i]);
        }

        return View<T>(m_Data, indices.size(), indices);
    }

    ~View() {}

private:
    const T* m_Data = nullptr;
    std::vector<size_t> m_Indices;
};

#endif
