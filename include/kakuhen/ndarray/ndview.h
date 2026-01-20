#pragma once
#include "kakuhen/ndarray/detail.h"
#include "kakuhen/ndarray/slice.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace kakuhen::ndarray {

template <typename T, typename S>
class NDArray;  // forward declaration

/*!
 * @brief A non-owning view into an NDArray.
 *
 * This class provides a view into a potentially multi-dimensional array
 * without owning the underlying data. It allows for flexible manipulation
 * and slicing of array data without copying the data itself.
 *
 * Optimization: Metadata (shape and strides) are stored in a single
 * contiguous memory block to minimize allocations.
 *
 * @tparam T The type of the elements in the array.
 * @tparam S The type of the indices and dimensions.
 */
template <typename T, typename S>
class NDView {
 public:
  using value_type = T;
  using size_type = S;

  /*!
   * @brief Default constructor.
   *
   * Creates an empty `NDView` with 0 dimensions and no data.
   */
  NDView()
      : ndim_(0),
        total_size_(0),
        metadata_block_(nullptr),
        shape_(nullptr),
        strides_(nullptr),
        data_(nullptr) {}

  /*!
   * @brief Constructs an `NDView` by copying shape and strides.
   *
   * Allocates a single block for metadata and copies the provided shape and strides.
   *
   * @param data A raw pointer to the start of the data.
   * @param shape Pointer to the shape array.
   * @param strides Pointer to the strides array.
   * @param ndim The number of dimensions.
   */
  NDView(T* data, const S* shape, const S* strides, S ndim) : ndim_(ndim), data_(data) {
    if (ndim_ > 0) {
      metadata_block_ = std::make_unique<S[]>(2 * ndim_);
      shape_ = metadata_block_.get();
      strides_ = metadata_block_.get() + ndim_;

      std::copy_n(shape, ndim_, shape_);
      std::copy_n(strides, ndim_, strides_);

      total_size_ = 1;
      for (S i = 0; i < ndim_; ++i) {
        total_size_ *= shape_[i];
      }
    } else {
      total_size_ = 0;
      shape_ = nullptr;
      strides_ = nullptr;
    }
  }

  /*!
   * @brief Implicit conversion constructor from an `NDArray`.
   *
   * This allows an `NDView` to be easily created from an `NDArray` instance,
   * effectively creating a view of the entire array.
   *
   * @param arr The `NDArray` to create a view of.
   */
  NDView(NDArray<T, S>& arr);  // only declaration; definition after NDArray

  /// @name Lifecycle
  /// @{
  NDView(NDView&& other) noexcept
      : ndim_(other.ndim_),
        total_size_(other.total_size_),
        metadata_block_(std::move(other.metadata_block_)),
        shape_(other.shape_),
        strides_(other.strides_),
        data_(other.data_) {
    other.ndim_ = 0;
    other.total_size_ = 0;
    other.shape_ = nullptr;
    other.strides_ = nullptr;
    other.data_ = nullptr;
  }

  NDView& operator=(NDView&& other) noexcept {
    if (this != &other) {
      ndim_ = other.ndim_;
      total_size_ = other.total_size_;
      metadata_block_ = std::move(other.metadata_block_);
      shape_ = other.shape_;
      strides_ = other.strides_;
      data_ = other.data_;

      other.ndim_ = 0;
      other.total_size_ = 0;
      other.shape_ = nullptr;
      other.strides_ = nullptr;
      other.data_ = nullptr;
    }
    return *this;
  }

  NDView(const NDView&) = delete;
  NDView& operator=(const NDView&) = delete;
  /// @}

  /*!
   * @brief Get the number of dimensions of the view.
   *
   * @return The number of dimensions.
   */
  [[nodiscard]] inline S ndim() const noexcept {
    return ndim_;
  }
  /*!
   * @brief Get the shape of the view.
   *
   * @return A span containing the size of each dimension.
   */
  [[nodiscard]] inline std::span<const S> shape() const noexcept {
    if (!shape_) return {};
    return {shape_, static_cast<size_t>(ndim_)};
  }

  /*!
   * @brief Get the strides of the view.
   *
   * @return A span containing the stride of each dimension.
   */
  [[nodiscard]] inline std::span<const S> strides() const noexcept {
    if (!strides_) return {};
    return {strides_, static_cast<size_t>(ndim_)};
  }

  /*!
   * @brief Get the total number of elements in the view.
   *
   * @return The total number of elements.
   */
  [[nodiscard]] inline S size() const noexcept {
    return total_size_;
  }
  /*!
   * @brief Check if the view is empty.
   *
   * @return True if the view is empty, false otherwise.
   */
  [[nodiscard]] inline bool empty() const noexcept {
    return total_size_ == 0;
  }

  [[nodiscard]] inline T* data() noexcept {
    return data_;
  }
  [[nodiscard]] inline const T* data() const noexcept {
    return data_;
  }

  template <typename... Indices>
  T& operator()(Indices... indices) noexcept {
    static_assert((std::is_integral_v<Indices> && ...), "All indices must be integral types");
    assert(sizeof...(indices) == ndim_);
    assert(((indices >= 0) && ...));
    S idx[] = {static_cast<S>(indices)...};
    return data_[detail::flat_index<S>(idx, strides_, shape_, ndim_)];
  }

  /*!
   * @brief Access an element of the view using multi-dimensional indices.
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
    return data_[detail::flat_index<S>(idx, strides_, shape_, ndim_)];
  }

  /*!
   * @brief Fills the entire view with a specified value.
   *
   * @param value The value to fill the view with.
   */
  void fill(const T& value) {
    if (!data_) return;
    std::fill(data_, data_ + total_size_, value);
  }

  /*!
   * @brief Creates a new `NDView` representing a slice of the current view.
   *
   * @param slices A vector of `Slice` objects, one for each dimension,
   * defining the slicing parameters.
   * @return A new `NDView` representing the specified slice.
   */
  [[nodiscard]] NDView<T, S> slice(const std::vector<Slice<S>>& slices) const {
    assert(slices.size() == ndim_);

    // Allocate single block for new metadata
    auto metadata = std::make_unique<S[]>(2 * ndim_);
    S* new_shape = metadata.get();
    S* new_strides = metadata.get() + ndim_;

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

    return NDView<T, S>(data_ + base_offset, std::move(metadata), ndim_);
  }

  /*!
   * @brief Creates a new `NDView` with a reshaped view of the underlying data.
   *
   * This operation requires the total number of elements to remain the same.
   * Currently, it only works on contiguous views.
   *
   * @param shape A vector specifying the new shape of the view.
   * @return A new `NDView` with the reshaped dimensions.
   * @throws std::runtime_error if the view is not contiguous or total size changes.
   */
  [[nodiscard]] NDView<T, S> reshape(const std::vector<S>& shape) const {
    S old_size = 1;
    for (S i = 0; i < ndim_; ++i)
      old_size *= shape_[i];

    S new_size = 1;
    for (auto s : shape)
      new_size *= s;

    assert(old_size == new_size);

#ifndef NDEBUG
    // check if view is contiguous (row-major layout)
    bool contiguous = true;
    S expected_stride = 1;
    for (S i = ndim_; i-- > 0;) {
      if (shape_[i] == 1) continue;  // skip size-1 dimensions
      if (strides_[i] != expected_stride) {
        contiguous = false;
        break;
      }
      expected_stride *= shape_[i];
    }
    assert(contiguous && "reshape only works on contiguous views");
#endif

    S new_ndim = static_cast<S>(shape.size());
    auto metadata = std::make_unique<S[]>(2 * new_ndim);
    S* new_shape = metadata.get();
    S* new_strides = metadata.get() + new_ndim;

    // Compute row-major strides for the new shape
    S stride = 1;
    for (S i = new_ndim; i-- > 0;) {
      new_shape[i] = shape[i];
      new_strides[i] = stride;
      stride *= shape[i];
    }

    return NDView<T, S>(data_, std::move(metadata), new_ndim);
  }

  /*!
   * @brief Extracts a diagonal from the view.
   *
   * This creates a new `NDView` that represents the diagonal elements
   * formed by two specified dimensions. The two dimensions must have
   * the same size.
   *
   * @param dim1 The first dimension to form the diagonal.
   * @param dim2 The second dimension to form the diagonal.
   * @return A new `NDView` representing the diagonal.
   * @throws std::runtime_error if `dim1` or `dim2` are out of bounds,
   * or if the shapes of `dim1` and `dim2` are not equal.
   */
  [[nodiscard]] NDView<T, S> diagonal(S dim1, S dim2) const {
    assert(dim1 >= 0 && dim1 < ndim_ && dim2 >= 0 && dim2 < ndim_);
    assert(shape_[dim1] == shape_[dim2]);  // must be square along those axes

    auto new_ndim = ndim_ - 1;
    auto metadata = std::make_unique<S[]>(2 * new_ndim);
    S* new_shape = metadata.get();
    S* new_strides = metadata.get() + new_ndim;

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

    return NDView<T, S>(data_, std::move(metadata), new_ndim);
  }

 private:
  /*!
   * @brief Internal constructor taking ownership of a metadata block.
   */
  NDView(T* data, std::unique_ptr<S[]> metadata, S ndim)
      : ndim_(ndim), metadata_block_(std::move(metadata)), data_(data) {
    if (ndim_ > 0 && metadata_block_) {
      shape_ = metadata_block_.get();
      strides_ = metadata_block_.get() + ndim_;

      total_size_ = 1;
      for (S i = 0; i < ndim_; ++i) {
        total_size_ *= shape_[i];
      }
    } else {
      total_size_ = 0;
      shape_ = nullptr;
      strides_ = nullptr;
    }
  }

  S ndim_;
  S total_size_;
  std::unique_ptr<S[]> metadata_block_;  // Holds shape + strides
  S* shape_ = nullptr;
  S* strides_ = nullptr;
  T* data_ = nullptr;

};  // class NDView

}  // namespace kakuhen::ndarray
