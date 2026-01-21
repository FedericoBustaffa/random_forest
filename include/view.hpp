#ifndef VIEW_HPP
#define VIEW_HPP

#include <vector>

template <typename T>
class View
{
public:
    View(const std::vector<T>& v, std::vector<size_t>& indices)
        : m_Data(v), m_Indices(indices)
    {
    }

    View(const View<T>& other, std::vector<size_t>& indices)
        : m_Data(other.m_Data), m_Indices(indices)
    {
    }

    inline size_t size() const { return m_Indices.size(); }

    inline bool empty() const { return m_Indices.empty(); }

    inline const std::vector<size_t>& indices() const { return m_Indices; }

    inline const T& operator[](size_t idx) const
    {
        return m_Data[m_Indices[idx]];
    }

    inline T& operator[](size_t idx) { return m_Data[m_Indices[idx]]; }

    ~View() {}

private:
    const std::vector<T>& m_Data;
    const std::vector<size_t>& m_Indices;
};

#endif
