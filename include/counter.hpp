#ifndef COUNTER_HPP
#define COUNTER_HPP

#include <cstddef>
#include <vector>

class Counter
{
public:
    Counter(size_t n_labels) : m_Counters(n_labels, 0) {}

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
