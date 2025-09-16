#pragma once
#include "kakuhen/ndarray/detail.h"
#include "kakuhen/ndarray/slice.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace kakuhen::ndarray {

template <typename T, typename S>
class NDView {
 public:
  using value_type = T;
  using size_type = S;

  NDView(T* data, std::unique_ptr<S[]> shape, std::unique_ptr<S[]> strides,
         S ndim)
      : data_(data),
        shape_(std::move(shape)),
        strides_(std::move(strides)),
        ndim_(ndim) {}

  inline S ndim() const noexcept {
    return ndim_;
  }
  inline std::vector<S> shape() const {
    return std::vector<S>(shape_.get(), shape_.get() + ndim_);
  }
  // inline const S* shape() const noexcept {
  //   return shape_.get();
  // }
  // inline const S* strides() const noexcept {
  //   return strides_.get();
  // }

  template <typename... Indices>
  T& operator()(Indices... indices) noexcept {
    static_assert((std::is_integral_v<Indices> && ...),
                  "All indices must be integral types");
    assert(sizeof...(indices) == ndim_);
    assert(((indices >= 0) && ...));
    S idx[] = {static_cast<S>(indices)...};
    return data_[detail::flat_index<S>(idx, strides_.get(), shape_.get(),
                                       ndim_)];
  }

  template <typename... Indices>
  const T& operator()(Indices... indices) const noexcept {
    static_assert((std::is_integral_v<Indices> && ...),
                  "All indices must be integral types");
    assert(sizeof...(indices) == ndim_);
    assert(((indices >= 0) && ...));
    S idx[] = {static_cast<S>(indices)...};
    return data_[detail::flat_index<S>(idx, strides_.get(), shape_.get(),
                                       ndim_)];
  }

  NDView<T, S> slice(const std::vector<Slice<S>>& slices) const {
    assert(slices.size() == ndim_);

    auto new_shape = std::make_unique<S[]>(ndim_);
    auto new_strides = std::make_unique<S[]>(ndim_);
    S base_offset = 0;

    for (S i = 0; i < ndim_; ++i) {
      const auto& s = slices[i];

      S begin = s.start.value_or(0);
      S end = s.stop.value_or(shape_[i]);
      S step = s.step.value_or(1);

      assert(begin >= 0 && begin <= end && end <= shape_[i] && step > 0);

      new_shape[i] = (end - begin + step - 1) / step;
      new_strides[i] = strides_[i] * step;
      base_offset += begin * strides_[i];
    }

    return NDView<T, S>(data_ + base_offset, std::move(new_shape),
                        std::move(new_strides), ndim_);
  }

 private:
  T* data_;
  std::unique_ptr<S[]> shape_;
  std::unique_ptr<S[]> strides_;
  S ndim_;
};  // class NDView

}  // namespace kakuhen::ndarray
