#include "vector.hpp"

#include <cstring>
#include <utility>

Vector::Vector(const double* data, size_t size) : VectorView(nullptr, size)
{
    m_Data = new double[size];
    std::memcpy(m_Data, data, size * sizeof(double));
    m_View = m_Data;
}

Vector::Vector(const Vector& other) : VectorView(nullptr, other.m_Size)
{
    m_Data = new double[m_Size];
    std::memcpy(m_Data, other.m_Data, m_Size * sizeof(double));
    m_View = m_Data;
}

Vector::Vector(Vector&& other) noexcept
    : VectorView(std::move(other)), m_Data(other.m_Data)
{
    m_View = m_Data;
    other.m_Data = nullptr;
}

Vector& Vector::operator=(const Vector& other)
{
    if (this != &other)
    {
        VectorView::operator=(other);

        delete[] m_Data;
        m_Data = new double[m_Size];
        std::memcpy(m_Data, other.m_Data, m_Size * sizeof(double));

        m_View = m_Data;
    }

    return *this;
}

Vector& Vector::operator=(Vector&& other) noexcept
{
    if (this != &other)
    {
        delete[] m_Data;

        VectorView::operator=(std::move(other));
        m_Data = other.m_Data;
        m_View = m_Data;

        other.m_Data = nullptr;
    }

    return *this;
}

Vector::~Vector() { delete[] m_Data; }
