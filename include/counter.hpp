#ifndef COUNTER_HPP
#define COUNTER_HPP

#include <cstddef>
#include <vector>

class Counter
{
public:
    Counter(size_t n_values) : m_Table(n_values, 0) {}

    ~Counter() {}

private:
    std::vector<size_t> m_Table;
};

#endif
