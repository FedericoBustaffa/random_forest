#include "tensor.hpp"

#include <cstring>
#include <stdexcept>

Tensor::Tensor(double scalar)
{
    m_Data = new double[1];
    *m_Data = scalar;
}

Tensor::Tensor(const double* data, const std::vector<size_t>& shape)
    : m_Shape(shape)
{
    m_Size = 1;
    for (const auto& d : shape)
        m_Size *= d;

    m_Stride = 1;
    for (size_t j = 1; j < m_Shape.size(); j++)
        m_Stride *= m_Shape[j];

    m_Data = new double[m_Size];
    std::memcpy(m_Data, data, m_Size * sizeof(double));
}

Tensor::Tensor(const Tensor& other)
    : m_Shape(other.m_Shape), m_Size(other.m_Size), m_Stride(other.m_Stride)
{
    m_Data = new double[m_Size];
    std::memcpy(m_Data, other.m_Data, m_Size * sizeof(double));
}

Tensor::Tensor(Tensor&& other)
    : m_Shape(other.m_Shape), m_Size(other.m_Size), m_Stride(other.m_Stride),
      m_Data(other.m_Data)
{
    other.m_Shape.clear();
    other.m_Size = 0;
    other.m_Stride = 0;
    other.m_Data = nullptr;
}

Tensor::operator double() const { return *m_Data; }

Tensor Tensor::operator[](size_t i) const
{
    // try to use [] on a scalar
    if (m_Shape.empty())
        throw std::runtime_error("scalar is not supscriptable");

    if (m_Shape.size() == 1)
        return Tensor(m_Data[i]); // scalar

    std::vector<size_t> shape(m_Shape.begin() + 1, m_Shape.end());

    std::vector<double> data;
    for (size_t j = 0; j < m_Shape[1]; j++)
        data.push_back(m_Data[i * m_Stride + j]);

    return Tensor(data.data(), shape);
}

Tensor::~Tensor() { delete[] m_Data; }
