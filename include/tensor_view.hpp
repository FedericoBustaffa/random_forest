#ifndef TENSOR_VIEW_HPP
#define TENSOR_VIEW_HPP

#include <vector>

class TensorView
{
public:
    TensorView(const double* data, const std::vector<size_t>& shape,
               const std::vector<size_t>& strides);

    TensorView(const double* data, const std::vector<size_t>& shape);

    TensorView(const TensorView& other);

    TensorView(TensorView&& other) noexcept;

    virtual inline const std::vector<size_t>& shape() const { return m_Shape; }

    virtual inline const size_t ndim() const { return m_Shape.size(); }

    virtual inline size_t size() const { return m_Size; }

    virtual inline operator double() const { return *m_View; }

    TensorView& operator=(const TensorView& other);

    TensorView& operator=(TensorView&& other) noexcept;

    virtual TensorView operator[](size_t idx) const;

    virtual TensorView operator()(size_t idx, size_t axis = 0) const;

    virtual ~TensorView();

protected:
    std::vector<size_t> m_Shape;
    std::vector<size_t> m_Strides;

    size_t m_Size;
    const double* m_View = nullptr;
};

#endif
