#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>

class Tensor
{
public:
    Tensor(const double* data, const std::vector<size_t>& shape);

    Tensor(double scalar);

    Tensor(const Tensor& other);

    Tensor(Tensor&& other);

    inline const std::vector<size_t>& shape() const { return m_Shape; }

    inline const size_t ndim() const { return m_Shape.size(); }

    inline size_t size() const { return m_Size; }

    void operator=(const Tensor& other);

    void operator=(Tensor&& other);

    inline operator double() const { return *m_Data; }

    Tensor operator[](size_t i) const;

    ~Tensor();

private:
    std::vector<size_t> m_Shape;
    size_t m_Size;
    std::vector<size_t> m_Strides;
    double* m_Data;
};

#endif
