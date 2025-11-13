#include "tensor.hpp"

#include <cstring>

Tensor::Tensor(const double* data, const std::vector<size_t>& shape)
    : TensorView(data, shape)
{
    m_Data = new double[m_Size];
    std::memcpy(m_Data, data, m_Size * sizeof(double));
}

Tensor::Tensor(const Tensor& other) : TensorView(other)
{
    m_Data = new double[m_Size];
    std::memcpy(m_Data, other.m_Data, m_Size * sizeof(double));
}

Tensor::Tensor(Tensor&& other) : TensorView(std::move(other))
{
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

Tensor::~Tensor() { delete[] m_Data; }
