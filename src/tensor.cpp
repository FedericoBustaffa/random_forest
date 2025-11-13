#include "tensor.hpp"

#include <cstring>

Tensor::Tensor(const double* data, const std::vector<size_t>& shape)
    : TensorView(nullptr, shape)
{
    m_Data = new double[m_Size];
    std::memcpy(m_Data, data, m_Size * sizeof(double));

    m_View = m_Data;
}

Tensor::Tensor(const Tensor& other) : TensorView(other)
{
    m_Data = new double[m_Size];
    std::memcpy(m_Data, other.m_Data, m_Size * sizeof(double));

    m_View = m_Data;
}

Tensor::Tensor(Tensor&& other) noexcept : TensorView(std::move(other))
{
    m_Data = other.m_Data;
    other.m_Data = nullptr;

    m_View = m_Data;
}

Tensor& Tensor::operator=(const Tensor& other)
{
    TensorView::operator=(other);

    delete[] m_Data;
    m_Data = new double[m_Size];
    std::memcpy(m_Data, other.m_Data, m_Size * sizeof(double));

    m_View = m_Data;

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    TensorView::operator=(std::move(other));

    delete[] m_Data;
    m_Data = other.m_Data;
    m_View = m_Data;

    other.m_Size = 0;
    other.m_Data = nullptr;
    other.m_View = nullptr;

    return *this;
}

Tensor::~Tensor() { delete[] m_Data; }
