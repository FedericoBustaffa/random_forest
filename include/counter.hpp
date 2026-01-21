#ifndef COUNTER_HPP
#define COUNTER_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "view.hpp"

class Counter
{
public:
    Counter(const View<uint32_t>& y)
    {
        size_t found = 0;
        for (size_t i = 0; i < y.size(); ++i)
        {
            if (y[i] >= found)
                found++;
        }

        m_Counters.resize(found, 0);
    }

    inline size_t size() const { return m_Counters.size(); }

    inline const size_t& operator[](size_t idx) const
    {
        return m_Counters[idx];
    }

    inline size_t& operator[](size_t idx) { return m_Counters[idx]; }

    inline void reset() {}

    ~Counter() {}

private:
    std::vector<size_t> m_Counters;
};

#endif
