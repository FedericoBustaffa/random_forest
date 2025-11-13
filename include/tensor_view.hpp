#ifndef TENSOR_VIEW_HPP
#define TENSOR_VIEW_HPP

#include <vector>

class TensorView
{
public:
    TensorView(const double* data, const std::vector<size_t>& shape);

    TensorView(const TensorView& other);

    virtual inline const std::vector<size_t>& shape() const { return m_Shape; }

    virtual inline const size_t ndim() const { return m_Shape.size(); }

    virtual inline size_t size() const { return m_Size; }

    virtual inline operator double() const { return *m_View; }

    void operator=(const TensorView& other);

    virtual TensorView operator[](size_t i) const;

    virtual ~TensorView();

protected:
    std::vector<size_t> m_Shape;
    size_t m_Size;
    std::vector<size_t> m_Strides;

private:
    const double* m_View;
};

#endif
