#pragma once
#include "kakuhen/ndarray/ndview.h"
#include "kakuhen/util/serialize.h"
#include "kakuhen/util/type.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace kakuhen::ndarray {

template <typename T,
          typename S = uint32_t>  // index up to 4'294'967'295 should suffice?
class NDArray {
 public:
  using value_type = T;
  using size_type = S;

  NDArray() : NDArray(0, {}) {}

  NDArray(S ndim, const S* shape)
      : ndim_(ndim), shape_(new S[ndim]), strides_(new S[ndim]) {
    for (S i = 0; i < ndim; ++i) {
      shape_[i] = shape[i];
    }
    compute_strides();
    data_ = std::make_unique<T[]>(total_size_);
  }

  NDArray(const std::vector<S>& shape) : NDArray(shape.size(), shape.data()) {}
  NDArray(std::initializer_list<S> shape) : NDArray(std::vector<S>(shape)) {}

  NDArray(NDArray&&) noexcept = default;
  NDArray& operator=(NDArray&&) noexcept = default;
  //> single ownership of the data
  NDArray(const NDArray&) = delete;
  NDArray& operator=(const NDArray&) = delete;

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
  inline S size() const noexcept {
    return total_size_;
  }

  inline T* data() noexcept {
    return data_.get();
  }
  inline const T* data() const noexcept {
    return data_.get();
  }

  /// Flat (linear) element access
  inline T& operator[](S idx) noexcept {
    assert(idx < total_size_);  // optional bound check
    return data_[idx];
  }

  inline const T& operator[](S idx) const noexcept {
    assert(idx >= 0 && idx < total_size_);
    return data_[idx];
  }

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

  void fill(const T& value) {
    std::fill(data_.get(), data_.get() + total_size_, value);
  }

  NDView<T, S> slice(const std::vector<Slice<S>>& slices) const {
    assert(slices.size() == static_cast<size_t>(ndim_));

    auto new_shape = std::make_unique<S[]>(ndim_);
    auto new_strides = std::make_unique<S[]>(ndim_);

    S base_offset = 0;

    for (S i = 0; i < ndim_; ++i) {
      const auto& s = slices[i];

      S begin = s.start.value_or(0);
      S end = s.stop.value_or(shape_[i]);
      S step = s.step.value_or(1);

      assert(begin >= 0 && begin < end && end <= shape_[i] && step > 0);

      S len = (end - begin + step - 1) / step;
      new_shape[i] = len;

      new_strides[i] = strides_[i] * step;
      base_offset += begin * strides_[i];
    }

    return NDView<T, S>(data_.get() + base_offset, std::move(new_shape),
                        std::move(new_strides), ndim_);
  }

  void serialize(std::ostream& out, bool with_type = false) const noexcept {
    if (with_type) {
      int16_t T_tos = kakuhen::util::type::get_type_or_size<T>();
      int16_t S_tos = kakuhen::util::type::get_type_or_size<S>();
      kakuhen::util::serialize::serialize_one<int16_t>(out, T_tos);
      kakuhen::util::serialize::serialize_one<int16_t>(out, S_tos);
    }
    kakuhen::util::serialize::serialize_one<S>(out, ndim_);
    kakuhen::util::serialize::serialize_array<S>(out, shape_.get(), ndim_);
    kakuhen::util::serialize::serialize_one<S>(out, total_size_);
    kakuhen::util::serialize::serialize_array<T>(out, data_.get(), total_size_);
  }

  void deserialize(std::istream& in, bool with_type = false) {
    if (with_type) {
      int16_t T_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, T_tos);
      if (T_tos != kakuhen::util::type::get_type_or_size<T>()) {
        throw std::runtime_error("type or size mismatch for typename T");
      }
      int16_t S_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, S_tos);
      if (S_tos != kakuhen::util::type::get_type_or_size<S>()) {
        throw std::runtime_error("type or size mismatch for typename S");
      }
    }
    kakuhen::util::serialize::deserialize_one<S>(in, ndim_);
    shape_ = std::make_unique<S[]>(ndim_);
    kakuhen::util::serialize::deserialize_array<S>(in, shape_.get(), ndim_);
    strides_ = std::make_unique<S[]>(ndim_);
    compute_strides();  // populates `strides_` & computes `total_size_`
    S total_size_in;
    kakuhen::util::serialize::deserialize_one<S>(in, total_size_in);
    assert(total_size_in == total_size_);
    kakuhen::util::serialize::deserialize_array<T>(in, data_.get(),
                                                   total_size_);
  }

 private:
  void compute_strides() {
    //> row-major layout
    S stride = 1;
    //> unsigned "underflows" to large positive number
    // for (S i = ndim_ - 1; i >= 0 && i < ndim_; --i) {
    for (S i = ndim_; i-- > 0;) {
      strides_[i] = stride;
      stride *= shape_[i];
    }
    total_size_ = stride;
  }

  S ndim_;
  std::unique_ptr<S[]> shape_;
  std::unique_ptr<S[]> strides_;
  S total_size_;
  std::unique_ptr<T[]> data_;
};  // class NDArray

}  // namespace kakuhen::ndarray
