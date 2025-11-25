#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "vector_view.hpp"

class Vector : public VectorView
{
public:
    Vector(size_t size);

    Vector(const double* data, size_t size);

    Vector(const Vector& other);

    Vector(Vector&& other) noexcept;

    Vector& operator=(const Vector& other);

    Vector& operator=(Vector&& other) noexcept;

    inline double operator[](size_t idx) const { return m_Data[idx]; }

    inline double& operator[](size_t idx) { return m_Data[idx]; }

    ~Vector();

private:
    double* m_Data;
};

#endif
