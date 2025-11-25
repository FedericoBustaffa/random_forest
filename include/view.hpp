#ifndef VIEW_HPP
#define VIEW_HPP

#include <cassert>
#include <vector>

#include "mask.hpp"

class View
{
public:
    View(const std::vector<double>& data) : m_Data(data), m_Indices(data.size())
    {
        for (size_t i = 0; i < data.size(); i++)
            m_Indices[i] = i;
    }

    View(const std::vector<double>& data, std::vector<size_t> indices)
        : m_Data(data), m_Indices(indices)
    {
    }

    inline size_t size() const { return m_Indices.size(); }

    double operator[](size_t idx) const
    {
        assert(idx < m_Indices.size());
        return m_Data[m_Indices[idx]];
    }

    View operator[](const std::vector<size_t>& indices) const
    {
        std::vector<size_t> new_indices(indices.size());
        for (size_t i = 0; i < indices.size(); i++)
            new_indices[i] = m_Indices[indices[i]];

        return View(m_Data, new_indices);
    }

    Mask operator<(double value) const
    {
        std::vector<bool> mask(m_Indices.size());
        for (size_t i = 0; i < m_Indices.size(); i++)
            mask[i] = m_Data[m_Indices[i]] < value;

        return Mask(mask);
    }

    View operator[](const Mask& mask) const
    {
        std::vector<size_t> indices;
        for (size_t i = 0; i < m_Indices.size(); i++)
        {
            if (mask[i])
                indices.push_back(m_Indices[i]);
        }

        return View(m_Data, indices);
    }

    ~View() {}

private:
    const std::vector<double>& m_Data;
    std::vector<size_t> m_Indices;
};

#endif
