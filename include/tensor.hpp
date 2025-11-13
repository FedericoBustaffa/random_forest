#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>

#include "tensor_view.hpp"

class Tensor : public TensorView
{
public:
    Tensor(const double* data, const std::vector<size_t>& shape);

    Tensor(const Tensor& other);

    Tensor(Tensor&& other);

    void operator=(const Tensor& other);

    void operator=(Tensor&& other);

    ~Tensor();

private:
    double* m_Data;
};

#endif
