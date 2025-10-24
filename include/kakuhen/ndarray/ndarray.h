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
  //> iterators
  using iterator = T*;
  using const_iterator = const T*;

  NDArray() : NDArray(0, {}) {}

  NDArray(S ndim, const S* shape) : ndim_(ndim), shape_(new S[ndim]), strides_(new S[ndim]) {
    for (S i = 0; i < ndim; ++i) {
      shape_[i] = shape[i];
    }
    compute_strides();
    data_ = std::make_unique<T[]>(total_size_);
  }

  NDArray(const std::vector<S>& shape) : NDArray(shape.size(), shape.data()) {}
  NDArray(std::initializer_list<S> shape) : NDArray(std::vector<S>(shape)) {}

  //> move
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
  inline bool empty() const noexcept {
    return total_size_ == 0;
  }

  inline T* data() noexcept {
    return data_.get();
  }
  inline const T* data() const noexcept {
    return data_.get();
  }

  /// Iterators
  inline T* begin() noexcept {
    return data_.get();
  }
  inline const T* begin() const noexcept {
    return data_.get();
  }
  inline const T* cbegin() const noexcept {
    return data_.get();
  }
  inline T* end() noexcept {
    return data_.get() + total_size_;
  }
  inline const T* end() const noexcept {
    return data_.get() + total_size_;
  }
  inline const T* cend() const noexcept {
    return data_.get() + total_size_;
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
    static_assert(sizeof...(Indices) > 0, "NDArray index operator must be called with at least one index");
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

  void fill(const T& value) {
    std::fill(data_.get(), data_.get() + total_size_, value);
  }

  inline NDView<T, S> view() const {
    auto shape_copy = std::make_unique<S[]>(ndim_);
    auto strides_copy = std::make_unique<S[]>(ndim_);

    std::copy_n(shape_.get(), ndim_, shape_copy.get());
    std::copy_n(strides_.get(), ndim_, strides_copy.get());

    return NDView<T, S>(data_.get(), std::move(shape_copy), std::move(strides_copy), ndim_);
  }

  inline NDView<T, S> slice(const std::vector<Slice<S>>& slices) const {
    return view().slice(slices);
  }

  inline NDView<T, S> reshape(const std::vector<S>& shape) const {
    return view().reshape(shape);
  }

  inline NDView<T, S> diagonal(S dim1, S dim2) const {
    return view().diagonal(dim1, dim2);
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
    data_ = std::make_unique<T[]>(total_size_);
    kakuhen::util::serialize::deserialize_array<T>(in, data_.get(), total_size_);
  }

 private:
  S ndim_;
  S total_size_;
  std::unique_ptr<S[]> shape_;
  std::unique_ptr<S[]> strides_;
  std::unique_ptr<T[]> data_;

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

};  // class NDArray

template <typename T, typename S>
NDView<T, S>::NDView(NDArray<T, S>& arr) : NDView(arr.view()) {}

}  // namespace kakuhen::ndarray
