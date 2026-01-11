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
#include <span>
#include <stdexcept>
#include <vector>

namespace kakuhen::ndarray {

/*!
 * @brief An owning multi-dimensional array class.
 *
 * This class provides a multi-dimensional array, similar to NumPy's ndarray.
 * It owns the underlying data, storing elements of a single type in a
 * contiguous block of memory. The data is managed with a `std::unique_ptr`,
 * so `NDArray` has unique ownership semantics (it cannot be copied, only moved).
 *
 * The array can be accessed using both flat (linear) indices and
 * multi-dimensional indices.
 *
 * @tparam T The type of the elements in the array.
 * @tparam S The type of the indices and dimensions.
 */
template <typename T,
          typename S = uint32_t>  // index up to 4'294'967'295 should suffice?
class NDArray {
 public:
  using value_type = T;
  using size_type = S;

  /// @name Iterators
  /// @{
  using iterator = T*;
  using const_iterator = const T*;
  /// @}

  /*!
   * @brief Default constructor.
   *
   * Creates an empty NDArray with 0 dimensions.
   */
  NDArray() : NDArray(0, {}) {}

  /*!
   * @brief Construct a new NDArray object.
   *
   * @param ndim The number of dimensions of the array.
   * @param shape An array containing the size of each dimension.
   */
  NDArray(S ndim, const S* shape) : ndim_(ndim), shape_(new S[ndim]), strides_(new S[ndim]) {
    for (S i = 0; i < ndim; ++i) {
      shape_[i] = shape[i];
    }
    compute_strides();
    data_ = std::make_unique_for_overwrite<T[]>(total_size_);
  }

  /*!
   * @brief Construct a new NDArray object.
   *
   * @param shape A vector containing the size of each dimension.
   */
  NDArray(const std::vector<S>& shape) : NDArray(shape.size(), shape.data()) {}
  /*!
   * @brief Construct a new NDArray object.
   *
   * @param shape An initializer list containing the size of each dimension.
   */
  NDArray(std::initializer_list<S> shape) : NDArray(std::vector<S>(shape)) {}

  /// @name Lifecycle
  /// @{
  NDArray(NDArray&&) noexcept = default;
  NDArray& operator=(NDArray&&) noexcept = default;

  NDArray(const NDArray&) = delete;
  NDArray& operator=(const NDArray&) = delete;
  /// @}

  /*!
   * @brief Get the number of dimensions of the array.
   *
   * @return The number of dimensions.
   */
  [[nodiscard]] inline S ndim() const noexcept {
    return ndim_;
  }
  /*!
   * @brief Get the shape of the array.
   *
   * @return A span containing the size of each dimension.
   */
  [[nodiscard]] inline std::span<const S> shape() const noexcept {
    return {shape_.get(), static_cast<size_t>(ndim_)};
  }

  /*!
   * @brief Get the strides of the array.
   *
   * @return A span containing the stride of each dimension.
   */
  [[nodiscard]] inline std::span<const S> strides() const noexcept {
    return {strides_.get(), static_cast<size_t>(ndim_)};
  }

  /*!
   * @brief Get the total number of elements in the array.
   *
   * @return The total number of elements.
   */
  [[nodiscard]] inline S size() const noexcept {
    return total_size_;
  }
  /*!
   * @brief Check if the array is empty.
   *
   * @return True if the array is empty, false otherwise.
   */
  [[nodiscard]] inline bool empty() const noexcept {
    return total_size_ == 0;
  }

  /*!
   * @brief Get a pointer to the underlying data.
   *
   * @return A pointer to the underlying data.
   */
  [[nodiscard]] inline T* data() noexcept {
    return data_.get();
  }
  /*!
   * @brief Get a const pointer to the underlying data.
   *
   * @return A const pointer to the underlying data.
   */
  [[nodiscard]] inline const T* data() const noexcept {
    return data_.get();
  }

  /// @name Iterators
  /// @{

  /*!
   * @brief Get an iterator to the beginning of the array.
   *
   * @return An iterator to the beginning of the array.
   */
  [[nodiscard]] inline T* begin() noexcept {
    return data_.get();
  }
  /*!
   * @brief Get a const iterator to the beginning of the array.
   *
   * @return A const iterator to the beginning of the array.
   */
  [[nodiscard]] inline const T* begin() const noexcept {
    return data_.get();
  }
  /*!
   * @brief Get a const iterator to the beginning of the array.
   *
   * @return A const iterator to the beginning of the array.
   */
  [[nodiscard]] inline const T* cbegin() const noexcept {
    return data_.get();
  }
  /*!
   * @brief Get an iterator to the end of the array.
   *
   * @return An iterator to the end of the array.
   */
  [[nodiscard]] inline T* end() noexcept {
    return data_.get() + total_size_;
  }
  /*!
   * @brief Get a const iterator to the end of the array.
   *
   * @return A const iterator to the end of the array.
   */
  [[nodiscard]] inline const T* end() const noexcept {
    return data_.get() + total_size_;
  }
  /*!
   * @brief Get a const iterator to the end of the array.
   *
   * @return A const iterator to the end of the array.
   */
  [[nodiscard]] inline const T* cend() const noexcept {
    return data_.get() + total_size_;
  }

  /// @}

  /// @name Flat (linear) element access
  /// @{

  /*!
   * @brief Access an element of the array using a flat index.
   *
   * @param idx The flat index of the element to access.
   * @return A reference to the element.
   */
  inline T& operator[](S idx) noexcept {
    assert(idx < total_size_);  // optional bound check
    return data_[idx];
  }

  /*!
   * @brief Access an element of the array using a flat index.
   *
   * @param idx The flat index of the element to access.
   * @return A const reference to the element.
   */
  inline const T& operator[](S idx) const noexcept {
    assert(idx >= 0 && idx < total_size_);
    return data_[idx];
  }

  /// @}

  /*!
   * @brief Access an element of the array using multi-dimensional indices.
   *
   * @tparam Indices The types of the indices.
   * @param indices The indices of the element to access.
   * @return A reference to the element.
   */
  template <typename... Indices>
  T& operator()(Indices... indices) noexcept {
    static_assert(sizeof...(Indices) > 0,
                  "NDArray index operator must be called with at least one index");
    static_assert((std::is_integral_v<Indices> && ...), "All indices must be integral types");
    assert(sizeof...(indices) == ndim_);
    assert(((indices >= 0) && ...));
    S idx[] = {static_cast<S>(indices)...};
    return data_[detail::flat_index<S>(idx, strides_.get(), shape_.get(), ndim_)];
  }

  /*!
   * @brief Access an element of the array using multi-dimensional indices.
   *
   * @tparam Indices The types of the indices.
   * @param indices The indices of the element to access.
   * @return A const reference to the element.
   */
  template <typename... Indices>
  const T& operator()(Indices... indices) const noexcept {
    static_assert((std::is_integral_v<Indices> && ...), "All indices must be integral types");
    assert(sizeof...(indices) == ndim_);
    assert(((indices >= 0) && ...));
    S idx[] = {static_cast<S>(indices)...};
    return data_[detail::flat_index<S>(idx, strides_.get(), shape_.get(), ndim_)];
  }

  /*!
   * @brief Fills the entire array with a specified value.
   *
   * @param value The value to fill the array with.
   */
  void fill(const T& value) {
    std::fill(data_.get(), data_.get() + total_size_, value);
  }

  /*!
   * @brief Creates a non-owning view of the entire array.
   *
   * @return An `NDView` that points to this array's data.
   */
  [[nodiscard]] inline NDView<T, S> view() const {
    auto shape_copy = std::make_unique<S[]>(ndim_);
    auto strides_copy = std::make_unique<S[]>(ndim_);

    std::copy_n(shape_.get(), ndim_, shape_copy.get());
    std::copy_n(strides_.get(), ndim_, strides_copy.get());

    return NDView<T, S>(data_.get(), std::move(shape_copy), std::move(strides_copy), ndim_);
  }

  /*!
   * @brief Creates a non-owning view representing a slice of the array.
   *
   * @param slices A vector of `Slice` objects defining the slicing parameters.
   * @return An `NDView` representing the specified slice.
   */
  [[nodiscard]] inline NDView<T, S> slice(const std::vector<Slice<S>>& slices) const {
    return view().slice(slices);
  }

  /*!
   * @brief Creates a non-owning view with a reshaped view of the array data.
   *
   * @param shape A vector specifying the new shape of the view.
   * @return An `NDView` with the reshaped dimensions.
   */
  [[nodiscard]] inline NDView<T, S> reshape(const std::vector<S>& shape) const {
    return view().reshape(shape);
  }

  /*!
   * @brief Extracts a diagonal from the array.
   *
   * @param dim1 The first dimension to form the diagonal.
   * @param dim2 The second dimension to form the diagonal.
   * @return An `NDView` representing the diagonal.
   */
  [[nodiscard]] inline NDView<T, S> diagonal(S dim1, S dim2) const {
    return view().diagonal(dim1, dim2);
  }

  /*!
   * @brief Serializes the array's metadata and data to a stream.
   *
   * @param out The output stream to write to.
   * @param with_type If true, includes type information in the serialization.
   */
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

  /*!
   * @brief Deserializes the array's metadata and data from a stream.
   *
   * @param in The input stream to read from.
   * @param with_type If true, expects and verifies type information in the stream.
   * @throws std::runtime_error if type information mismatches when `with_type` is true.
   */
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
    // row-major layout
    S stride = 1;
    // unsigned "underflows" to large positive number
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
