#ifndef MASK_HPP
#define MASK_HPP

#include <vector>

class Mask
{
public:
    Mask(const std::vector<bool>& data) : m_Data(data) {}

    inline bool operator[](size_t idx) const { return m_Data[idx]; }

    Mask operator!() const
    {
        std::vector<bool> mask(m_Data.size());
        for (size_t i = 0; i < m_Data.size(); i++)
            mask[i] = !m_Data[i];

        return Mask(mask);
    }

private:
    std::vector<bool> m_Data;
};

#endif
