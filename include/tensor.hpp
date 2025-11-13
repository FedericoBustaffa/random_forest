#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>

#include "tensor_view.hpp"

class Tensor : public TensorView
{
public:
    Tensor(const double* data, const std::vector<size_t>& shape);

    Tensor(const Tensor& other);

    Tensor(Tensor&& other) noexcept;

    Tensor& operator=(const Tensor& other);

    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

private:
    double* m_Data;
};

#endif
