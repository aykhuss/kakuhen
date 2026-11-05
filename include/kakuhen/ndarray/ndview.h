#pragma once
#include "kakuhen/ndarray/detail.h"
#include "kakuhen/ndarray/slice.h"
#include "kakuhen/util/small_vector.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

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
 * Shape and strides live inline up to rank 4 and on the heap above that, so
 * views up to rank 4 never allocate. Like `std::span`, the element constness
 * comes from `T` (use `const T` for read-only views) and the elements must
 * outlive the view.
 *
 * @tparam T The type of the elements in the array.
 * @tparam S The type of the indices and dimensions.
 */
template <typename T, typename S>
class NDView {
  // strides followed by shape
  using Metadata = kakuhen::util::SmallVector<S, 8>;
  template <typename, typename>
  friend class NDView;

 public:
  using value_type = T;
  using size_type = S;

  /*!
   * @brief Default constructor.
   *
   * Creates an empty `NDView` with 0 dimensions and no data.
   */
  NDView() = default;

  /*!
   * @brief Constructs an `NDView` by copying shape and strides.
   *
   * @param data A raw pointer to the start of the data.
   * @param shape Pointer to the shape array.
   * @param strides Pointer to the strides array.
   * @param ndim The number of dimensions.
   */
  NDView(T* data, const S* shape, const S* strides, S ndim) : data_(data) {
    metadata_.resize(2 * static_cast<std::size_t>(ndim));
    std::copy_n(strides, ndim, metadata_.data());
    std::copy_n(shape, ndim, metadata_.data() + ndim);
  }

  /*!
   * @brief Implicit conversion constructor from an `NDArray`.
   *
   * This allows an `NDView` to be easily created from an `NDArray` instance,
   * effectively creating a view of the entire array. A const array only
   * converts to a view of const elements.
   *
   * @param arr The `NDArray` to create a view of.
   */
  NDView(NDArray<std::remove_const_t<T>, S>& arr);  // defined after NDArray
  NDView(const NDArray<std::remove_const_t<T>, S>& arr)
    requires std::is_const_v<T>;

  /// add element constness; copies or moves the metadata, never the elements
  template <typename U>
    requires(std::is_same_v<T, const U> && !std::is_const_v<U>)
  NDView(NDView<U, S> other) noexcept : metadata_(std::move(other.metadata_)), data_(other.data_) {}

  /*!
   * @brief Get the number of dimensions of the view.
   *
   * @return The number of dimensions.
   */
  [[nodiscard]] inline S ndim() const noexcept {
    return static_cast<S>(metadata_.size() / 2);
  }
  /*!
   * @brief Get the shape of the view.
   *
   * @return A span containing the size of each dimension.
   */
  [[nodiscard]] inline std::span<const S> shape() const noexcept {
    return {shape_ptr(), static_cast<std::size_t>(ndim())};
  }

  /*!
   * @brief Get the strides of the view.
   *
   * @return A span containing the stride of each dimension.
   */
  [[nodiscard]] inline std::span<const S> strides() const noexcept {
    return {strides_ptr(), static_cast<std::size_t>(ndim())};
  }

  /*!
   * @brief Get the total number of elements in the view.
   *
   * @return The total number of elements.
   */
  [[nodiscard]] inline S size() const noexcept {
    return element_count(shape());
  }
  /*!
   * @brief Check if the view is empty.
   *
   * @return True if the view is empty, false otherwise.
   */
  [[nodiscard]] inline bool empty() const noexcept {
    return size() == 0;
  }

  /**
   * @brief Access the raw data pointer.
   * @return A pointer to the underlying data; `const` only if `T` is.
   */
  [[nodiscard]] inline T* data() const noexcept {
    return data_;
  }

  /**
   * @brief Access an element of the view using multi-dimensional indices.
   *
   * @tparam Indices The types of the indices.
   * @param indices The indices of the element to access.
   * @return A reference to the element; `const` only if `T` is.
   */
  template <typename... Indices>
  T& operator()(Indices... indices) const noexcept {
    static_assert((std::is_integral_v<Indices> && ...), "All indices must be integral types");
    constexpr std::size_t N = sizeof...(Indices);
    assert(N == ndim());
    assert(((indices >= 0) && ...));
    const auto idx = std::make_tuple(static_cast<S>(indices)...);
#ifndef NDEBUG
    S idx_arr[] = {static_cast<S>(indices)...};
    assert(detail::flat_index<S>(idx_arr, strides_ptr(), shape_ptr(), ndim()) ==
           detail::flat_index_unrolled<S>(idx, strides_ptr(), std::make_index_sequence<N>{}));
#endif
    return data_[detail::flat_index_unrolled<S>(idx, strides_ptr(), std::make_index_sequence<N>{})];
  }

  /*!
   * @brief Fills the elements of the view with a specified value.
   *
   * Only the elements selected by the shape and strides are written.
   *
   * @param value The value to fill the view with.
   */
  void fill(const T& value) const
    requires(!std::is_const_v<T>)
  {
    if (empty()) return;
    // merge the contiguous trailing axes into one run; size-1 axes never move the address
    const S* shape = shape_ptr();
    const S* strides = strides_ptr();
    S outer_rank = ndim();
    S run_size = 1;
    while (outer_rank > 0) {
      const S axis = outer_rank - 1;
      if (shape[axis] != 1 && strides[axis] != run_size) break;
      run_size *= shape[axis];
      --outer_rank;
    }
    fill_runs(data_, 0, outer_rank, run_size, value);
  }

  /*!
   * @brief Creates a new `NDView` representing a slice of the current view.
   *
   * @param slices A range of `Slice` objects, one for each dimension,
   * defining the slicing parameters.
   * @return A new `NDView` representing the specified slice.
   */
  [[nodiscard]] NDView slice(std::span<const Slice<S>> slices) const {
    const S rank = ndim();
    assert(slices.size() == static_cast<std::size_t>(rank));

    Metadata metadata;
    metadata.resize(2 * static_cast<std::size_t>(rank));
    S* new_strides = metadata.data();
    S* new_shape = metadata.data() + rank;

    S base_offset = 0;

    for (S i = 0; i < rank; ++i) {
      const auto& s = slices[static_cast<std::size_t>(i)];

      S begin = s.start.value_or(0);
      S end = s.stop.value_or(shape_ptr()[i]);
      S step = s.step.value_or(1);

      assert(begin >= 0 && begin <= end && end <= shape_ptr()[i] && step > 0);

      new_shape[i] = (end - begin + step - 1) / step;
      new_strides[i] = strides_ptr()[i] * step;
      base_offset += begin * strides_ptr()[i];
    }

    return NDView(data_ + base_offset, std::move(metadata));
  }
  [[nodiscard]] NDView slice(std::initializer_list<Slice<S>> slices) const {
    return slice(std::span(slices));
  }

  /*!
   * @brief Creates a new `NDView` with a reshaped view of the underlying data.
   *
   * This operation requires the total number of elements to remain the same.
   * Currently, it only works on contiguous views.
   *
   * @param shape A range specifying the new shape of the view.
   * @return A new `NDView` with the reshaped dimensions.
   * @pre The view is contiguous and the total size is unchanged (debug assertions).
   */
  [[nodiscard]] NDView reshape(std::span<const S> shape) const {
    assert(size() == element_count(shape));

#ifndef NDEBUG
    // check if view is contiguous (row-major layout)
    bool contiguous = true;
    S expected_stride = 1;
    for (S i = ndim(); i-- > 0;) {
      if (shape_ptr()[i] == 1) continue;  // skip size-1 dimensions
      if (strides_ptr()[i] != expected_stride) {
        contiguous = false;
        break;
      }
      expected_stride *= shape_ptr()[i];
    }
    assert(contiguous && "reshape only works on contiguous views");
#endif

    S new_ndim = static_cast<S>(shape.size());
    Metadata metadata;
    metadata.resize(2 * static_cast<std::size_t>(new_ndim));
    S* new_strides = metadata.data();
    S* new_shape = metadata.data() + new_ndim;

    // Compute row-major strides for the new shape
    S stride = 1;
    for (S i = new_ndim; i-- > 0;) {
      new_shape[i] = shape[static_cast<std::size_t>(i)];
      new_strides[i] = stride;
      stride *= shape[static_cast<std::size_t>(i)];
    }

    return NDView(data_, std::move(metadata));
  }
  [[nodiscard]] NDView reshape(std::initializer_list<S> shape) const {
    return reshape(std::span(shape));
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
   * @pre The dimensions are distinct, in bounds and equally sized.
   */
  [[nodiscard]] NDView diagonal(S dim1, S dim2) const {
    const S rank = ndim();
    assert(dim1 >= 0 && dim1 < rank && dim2 >= 0 && dim2 < rank && dim1 != dim2);
    assert(shape_ptr()[dim1] == shape_ptr()[dim2]);  // must be square along those axes

    const S new_ndim = rank - 1;
    Metadata metadata;
    metadata.resize(2 * static_cast<std::size_t>(new_ndim));
    S* new_strides = metadata.data();
    S* new_shape = metadata.data() + new_ndim;

    // Fill new shape/strides
    S idx = 0;
    for (S i = 0; i < rank; ++i) {
      if (i == dim1) {
        // collapse dim1 and dim2 into one
        new_shape[idx] = shape_ptr()[dim1];  // same size as either one
        new_strides[idx] = strides_ptr()[dim1] + strides_ptr()[dim2];
        ++idx;
      } else if (i == dim2) {
        continue;  // skip the second diagonal dim
      } else {
        new_shape[idx] = shape_ptr()[i];
        new_strides[idx] = strides_ptr()[i];
        ++idx;
      }
    }

    return NDView(data_, std::move(metadata));
  }

 private:
  NDView(T* data, Metadata metadata) noexcept : metadata_(std::move(metadata)), data_(data) {}

  // product of the extents; a rank-0 shape holds no elements
  [[nodiscard]] static S element_count(std::span<const S> extents) noexcept {
    if (extents.empty()) return 0;
    S total = 1;
    for (const S extent : extents)
      total *= extent;
    return total;
  }

  // write `run_size` elements at every offset spanned by the axes [axis, outer_rank)
  void fill_runs(T* base, S axis, S outer_rank, S run_size, const T& value) const {
    if (axis == outer_rank) {
      std::fill_n(base, run_size, value);
      return;
    }
    const S extent = shape_ptr()[axis];
    const S stride = strides_ptr()[axis];
    for (S i = 0; i < extent; ++i)
      fill_runs(base + i * stride, axis + 1, outer_rank, run_size, value);
  }

  [[nodiscard]] const S* strides_ptr() const noexcept {
    return metadata_.data();
  }
  [[nodiscard]] const S* shape_ptr() const noexcept {
    return metadata_.data() + ndim();
  }

  Metadata metadata_;
  T* data_ = nullptr;

};  // class NDView

// the owner constructors hide T inside remove_const_t, so deduction needs guides
template <typename T, typename S>
NDView(NDArray<T, S>&) -> NDView<T, S>;

template <typename T, typename S>
NDView(const NDArray<T, S>&) -> NDView<const T, S>;

}  // namespace kakuhen::ndarray
