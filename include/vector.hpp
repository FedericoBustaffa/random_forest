#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "vector_view.hpp"

class Vector : public VectorView
{
public:
    Vector(const double* data, size_t size);

    Vector(const Vector& other);

    Vector(Vector&& other) noexcept;

    Vector& operator=(const Vector& other);

    Vector& operator=(Vector&& other) noexcept;

    ~Vector();

private:
    double* m_Data;
};

#endif
