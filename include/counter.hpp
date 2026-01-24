#ifndef COUNTER_HPP
#define COUNTER_HPP

#include <cstddef>
#include <vector>

class Counter
{
public:
    Counter(size_t n_labels) : m_Counters(n_labels, 0) {}

    inline size_t size() const { return m_Counters.size(); }

    inline size_t total() const
    {
        size_t s = 0;
        for (size_t i = 0; i < m_Counters.size(); i++)
            s += m_Counters[i];

        return s;
    }

    inline const size_t& operator[](size_t idx) const
    {
        return m_Counters[idx];
    }

    inline size_t& operator[](size_t idx) { return m_Counters[idx]; }

    inline void reset() { std::fill(m_Counters.begin(), m_Counters.end(), 0); }

    ~Counter() {}

private:
    std::vector<size_t> m_Counters;
};

#endif
