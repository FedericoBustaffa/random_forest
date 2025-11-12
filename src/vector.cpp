#include "vector.hpp"

#include <cstring>

Vector::Vector(const double* data, size_t size) : m_Size(size)
{
    m_Data = new double[size];
    std::memcpy(m_Data, data, size * sizeof(double));
}

Vector::Vector(const Vector& other) : m_Size(other.m_Size)
{
    m_Data = new double[m_Size];
    std::memcpy(m_Data, other.m_Data, m_Size * sizeof(double));
}

Vector::Vector(Vector&& other) : m_Size(other.m_Size), m_Data(other.m_Data)
{
    other.m_Size = 0;
    other.m_Data = nullptr;
}

Vector::~Vector() { delete[] m_Data; }
