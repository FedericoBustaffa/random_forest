#ifndef MASK_HPP
#define MASK_HPP

#include <vector>

class Mask
{
public:
    Mask(const std::vector<bool>& data) : m_Data(data) {}

    Mask operator!() const
    {
        std::vector<bool> not_mask(m_Data.size());
        for (size_t i = 0; i < m_Data.size(); i++)
            not_mask[i] = !m_Data[i];

        return Mask(not_mask);
    }

    inline bool operator[](size_t idx) const { return m_Data[idx]; }

    inline size_t size() const { return m_Data.size(); }

private:
    std::vector<bool> m_Data;
};

#endif
