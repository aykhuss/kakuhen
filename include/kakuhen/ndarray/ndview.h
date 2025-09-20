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

  NDView() : data_(nullptr), shape_(nullptr), strides_(nullptr), ndim_(0) {}

  NDView(T* data, std::unique_ptr<S[]> shape, std::unique_ptr<S[]> strides, S ndim)
      : data_(data), shape_(std::move(shape)), strides_(std::move(strides)), ndim_(ndim) {}

  /// move
  NDView(NDView&&) noexcept = default;
  NDView& operator=(NDView&&) noexcept = default;

  /// no copy
  NDView(const NDView&) = delete;
  NDView& operator=(const NDView&) = delete;

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

  // inline const T* data() const noexcept {
  //   return data_;
  // }

  template <typename... Indices>
  T& operator()(Indices... indices) noexcept {
    static_assert((std::is_integral_v<Indices> && ...), "All indices must be integral types");
    assert(sizeof...(indices) == ndim_);
    assert(((indices >= 0) && ...));
    S idx[] = {static_cast<S>(indices)...};
    return data_[detail::flat_index<S>(idx, strides_.get(), shape_.get(), ndim_)];
  }

  template <typename... Indices>
  const T& operator()(Indices... indices) const noexcept {
    static_assert((std::is_integral_v<Indices> && ...), "All indices must be integral types");
    assert(sizeof...(indices) == ndim_);
    assert(((indices >= 0) && ...));
    S idx[] = {static_cast<S>(indices)...};
    return data_[detail::flat_index<S>(idx, strides_.get(), shape_.get(), ndim_)];
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

    return NDView<T, S>(data_ + base_offset, std::move(new_shape), std::move(new_strides), ndim_);
  }

  NDView<T, S> reshape(const std::vector<S>& shape) const {
    S old_size = 1;
    for (S i = 0; i < ndim_; ++i)
      old_size *= shape_[i];

    S new_size = 1;
    for (auto s : shape)
      new_size *= s;

    assert(old_size == new_size);

    auto new_shape = std::make_unique<S[]>(shape.size());
    auto new_strides = std::make_unique<S[]>(shape.size());

    // Compute row-major strides for the new shape
    S stride = 1;
    for (S i = shape.size(); i-- > 0;) {
      new_shape[i] = shape[i];
      new_strides[i] = stride;
      stride *= shape[i];
    }

    return NDView<T, S>(data_, std::move(new_shape), std::move(new_strides),
                        static_cast<S>(shape.size()));
  }

  NDView<T, S> diagonal(S dim1, S dim2) const {
    assert(dim1 >= 0 && dim1 < ndim_ && dim2 >= 0 && dim2 < ndim_);
    assert(shape_[dim1] == shape_[dim2]);  // must be square along those axes

    auto new_ndim = ndim_ - 1;
    auto new_shape = std::make_unique<S[]>(new_ndim);
    auto new_strides = std::make_unique<S[]>(new_ndim);

    // Fill new shape/strides
    S idx = 0;
    for (S i = 0; i < ndim_; ++i) {
      if (i == dim1) {
        // collapse dim1 and dim2 into one
        new_shape[idx] = shape_[dim1];  // same size as either one
        new_strides[idx] = strides_[dim1] + strides_[dim2];
        ++idx;
      } else if (i == dim2) {
        continue;  // skip the second diagonal dim
      } else {
        new_shape[idx] = shape_[i];
        new_strides[idx] = strides_[i];
        ++idx;
      }
    }

    return NDView<T, S>(data_, std::move(new_shape), std::move(new_strides), new_ndim);
  }

 private:
  T* data_;
  std::unique_ptr<S[]> shape_;
  std::unique_ptr<S[]> strides_;
  S ndim_;
};  // class NDView

}  // namespace kakuhen::ndarray
