#include "tensor.hpp"

#include <cstring>
#include <stdexcept>

Tensor::Tensor(const double* data, const std::vector<size_t>& shape)
    : m_Shape(shape), m_Strides(shape.size())
{
    // compute the total size of the tensor
    m_Size = 1;
    for (const auto& d : shape)
        m_Size *= d;

    m_Strides.back() = 1;

    for (int j = m_Strides.size() - 2; j >= 0; j--)
        m_Strides[j] = m_Strides[j + 1] * m_Shape[j + 1];

    m_Data = new double[m_Size];
    std::memcpy(m_Data, data, m_Size * sizeof(double));
}

Tensor::Tensor(double scalar) : m_Shape({}), m_Size(1), m_Strides({})
{
    m_Data = new double[1];
    m_Data[0] = scalar;
}

Tensor::Tensor(const Tensor& other)
    : m_Shape(other.m_Shape), m_Size(other.m_Size), m_Strides(other.m_Strides)
{
    m_Data = new double[m_Size];
    std::memcpy(m_Data, other.m_Data, m_Size * sizeof(double));
}

Tensor::Tensor(Tensor&& other)
    : m_Shape(std::move(other.m_Shape)), m_Size(other.m_Size),
      m_Strides(std::move(other.m_Strides)), m_Data(other.m_Data)
{
    other.m_Size = 0;
    other.m_Data = nullptr;
}

void Tensor::operator=(const Tensor& other)
{
    m_Shape = other.m_Shape;
    m_Size = other.m_Size;
    m_Strides = other.m_Strides;

    delete[] m_Data;
    m_Data = new double[m_Size];
    std::memcpy(m_Data, other.m_Data, m_Size * sizeof(double));
}

void Tensor::operator=(Tensor&& other)
{
    m_Shape = std::move(other.m_Shape);
    m_Size = other.m_Size;
    m_Strides = std::move(other.m_Strides);

    delete[] m_Data;
    m_Data = other.m_Data;

    other.m_Size = 0;
    other.m_Data = nullptr;
}

Tensor Tensor::operator[](size_t i) const
{
    // try to use [] on a scalar
    if (m_Shape.empty())
        throw std::runtime_error("scalar is not supscriptable");

    if (m_Shape.size() == 1)
        return Tensor(m_Data[i]); // scalar

    std::vector<size_t> shape(m_Shape.begin() + 1, m_Shape.end());
    std::vector<size_t> strides(m_Strides.begin() + 1, m_Strides.end());

    const double* data = m_Data + i * m_Strides[0];

    return Tensor(data, shape);
}

Tensor::~Tensor() { delete[] m_Data; }
