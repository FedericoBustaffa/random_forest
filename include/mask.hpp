#ifndef MASK_HPP
#define MASK_HPP

#include <vector>

class Mask
{
public:
    Mask(bool value, size_t size) : m_Data(size, value) {}

    Mask(const std::vector<bool>& data) : m_Data(data) {}

    Mask operator!() const
    {
        std::vector<bool> not_mask(m_Data.size());
        for (size_t i = 0; i < m_Data.size(); i++)
            not_mask[i] = !m_Data[i];

        return Mask(not_mask);
    }

    Mask operator&(const Mask& other) const
    {
        size_t n = m_Data.size();

        std::vector<bool> result(n);
        for (size_t i = 0; i < n; i++)
            result[i] = m_Data[i] && other[i];

        return Mask(result);
    }

    inline bool operator[](size_t idx) const { return m_Data[idx]; }

    inline size_t size() const { return m_Data.size(); }

private:
    std::vector<bool> m_Data;
};

#endif
