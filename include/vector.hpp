#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cstddef>

class Vector
{
public:
    Vector(const double* data, size_t size);

    Vector(const Vector& other);

    Vector(Vector&& other);

    inline size_t size() const { return m_Size; }

    inline double& operator[](size_t i) { return m_Data[i]; }

    inline double operator[](size_t i) const { return m_Data[i]; }

    ~Vector();

private:
    size_t m_Size;
    double* m_Data;
};

#endif
