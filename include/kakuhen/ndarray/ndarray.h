#pragma once
#include "kakuhen/ndarray/ndview.h"
#include "kakuhen/util/serialize.h"
#include "kakuhen/util/type.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace kakuhen::ndarray {

/*!
 * @brief An owning multi-dimensional array class.
 *
 * This class provides a multi-dimensional array, similar to NumPy's ndarray.
 * It uses a **Single Allocation Strategy** where metadata (shape, strides)
 * and the data elements are stored in a single contiguous memory block.
 * This minimizes heap allocations and improves cache locality.
 *
 * The array owns the data and is movable but not copyable to prevent
 * accidental deep copies of large datasets.
 *
 * @tparam T The type of the elements in the array.
 * @tparam S The type of the indices and dimensions.
 */
template <typename T, typename S = uint32_t>
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
   * Creates an empty NDArray with 0 dimensions and no allocated memory.
   */
  NDArray() : ndim_(0), total_size_(0), memory_block_(nullptr) {}

  /*!
   * @brief Construct a new NDArray from dimensions and a shape array.
   *
   * @param ndim The number of dimensions of the array.
   * @param shape A pointer to an array containing the size of each dimension.
   */
  NDArray(S ndim, const S* shape) {
    init(ndim, shape);
  }

  /*!
   * @brief Construct a new NDArray from a vector of shapes.
   *
   * @param shape A vector containing the size of each dimension.
   */
  NDArray(const std::vector<S>& shape) : NDArray(static_cast<S>(shape.size()), shape.data()) {}

  /*!
   * @brief Construct a new NDArray from an initializer list of shapes.
   *
   * @param shape An initializer list containing the size of each dimension.
   */
  NDArray(std::initializer_list<S> shape)
      : NDArray(static_cast<S>(shape.size()), std::data(shape)) {}

  /*!
   * @brief Construct a new NDArray from a span of shapes.
   *
   * @param shape A span containing the size of each dimension.
   */
  NDArray(std::span<const S> shape) : NDArray(static_cast<S>(shape.size()), shape.data()) {}

  /*!
   * @brief Destructor.
   *
   * Manually destructs elements if T is not trivially destructible.
   * The memory block is automatically released by the unique_ptr.
   */
  ~NDArray() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      if (data_) {
        std::destroy_n(data_, total_size_);
      }
    }
  }

  /// @name Lifecycle
  /// @{

  /*!
   * @brief Move constructor.
   *
   * Transfers ownership of the memory block and metadata from `other` to `this`.
   * `other` is left in an empty/reset state.
   *
   * @param other The array to move from.
   */
  NDArray(NDArray&& other) noexcept
      : ndim_(other.ndim_),
        total_size_(other.total_size_),
        memory_block_(std::move(other.memory_block_)),
        shape_(other.shape_),
        strides_(other.strides_),
        data_(other.data_) {
    other.ndim_ = 0;
    other.total_size_ = 0;
    other.shape_ = nullptr;
    other.strides_ = nullptr;
    other.data_ = nullptr;
  }

  /*!
   * @brief Move assignment operator.
   *
   * Replaces the current contents with those of `other`.
   * The previous data is destructed (if needed) and memory released.
   *
   * @param other The array to move from.
   * @return A reference to `this`.
   */
  NDArray& operator=(NDArray&& other) noexcept {
    if (this != &other) {
      // Destroy current elements if non-trivial
      if constexpr (!std::is_trivially_destructible_v<T>) {
        if (data_) {
          std::destroy_n(data_, total_size_);
        }
      }

      ndim_ = other.ndim_;
      total_size_ = other.total_size_;
      memory_block_ = std::move(other.memory_block_);
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

  // Disable copy semantics to enforce unique ownership
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
    if (!shape_) return {};
    return {shape_, static_cast<size_t>(ndim_)};
  }

  /*!
   * @brief Get the strides of the array.
   *
   * @return A span containing the stride of each dimension.
   */
  [[nodiscard]] inline std::span<const S> strides() const noexcept {
    if (!strides_) return {};
    return {strides_, static_cast<size_t>(ndim_)};
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
   * @brief Check if the array is empty (has zero total elements).
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
    return data_;
  }

  /*!
   * @brief Get a const pointer to the underlying data.
   *
   * @return A const pointer to the underlying data.
   */
  [[nodiscard]] inline const T* data() const noexcept {
    return data_;
  }

  /// @name Iterators
  /// @{

  /*!
   * @brief Get an iterator to the beginning of the array.
   *
   * @return An iterator to the beginning of the array.
   */
  [[nodiscard]] inline T* begin() noexcept {
    return data_;
  }
  /*!
   * @brief Get a const iterator to the beginning of the array.
   *
   * @return A const iterator to the beginning of the array.
   */
  [[nodiscard]] inline const T* begin() const noexcept {
    return data_;
  }
  /*!
   * @brief Get a const iterator to the beginning of the array.
   *
   * @return A const iterator to the beginning of the array.
   */
  [[nodiscard]] inline const T* cbegin() const noexcept {
    return data_;
  }
  /*!
   * @brief Get an iterator to the end of the array.
   *
   * @return An iterator to the end of the array.
   */
  [[nodiscard]] inline T* end() noexcept {
    return data_ + total_size_;
  }
  /*!
   * @brief Get a const iterator to the end of the array.
   *
   * @return A const iterator to the end of the array.
   */
  [[nodiscard]] inline const T* end() const noexcept {
    return data_ + total_size_;
  }
  /*!
   * @brief Get a const iterator to the end of the array.
   *
   * @return A const iterator to the end of the array.
   */
  [[nodiscard]] inline const T* cend() const noexcept {
    return data_ + total_size_;
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
    assert(idx >= 0 && idx < total_size_);
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
    constexpr std::size_t N = sizeof...(Indices);
    assert(N == ndim_);
    assert(((indices >= 0) && ...));
    const auto idx = std::make_tuple(static_cast<S>(indices)...);
#ifndef NDEBUG
    S idx_arr[] = {static_cast<S>(indices)...};
    assert(detail::flat_index<S>(idx_arr, strides_, shape_, ndim_) ==
           detail::flat_index_unrolled<S>(idx, strides_, std::make_index_sequence<N>{}));
#endif
    return data_[detail::flat_index_unrolled<S>(idx, strides_, std::make_index_sequence<N>{})];
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
    constexpr std::size_t N = sizeof...(Indices);
    assert(N == ndim_);
    assert(((indices >= 0) && ...));
    const auto idx = std::make_tuple(static_cast<S>(indices)...);
#ifndef NDEBUG
    S idx_arr[] = {static_cast<S>(indices)...};
    assert(detail::flat_index<S>(idx_arr, strides_, shape_, ndim_) ==
           detail::flat_index_unrolled<S>(idx, strides_, std::make_index_sequence<N>{}));
#endif
    return data_[detail::flat_index_unrolled<S>(idx, strides_, std::make_index_sequence<N>{})];
  }

  /*!
   * @brief Fills the entire array with a specified value.
   *
   * @param value The value to fill the array with.
   */
  void fill(const T& value) {
    if (!data_) return;
    std::fill(data_, data_ + total_size_, value);
  }

  /*!
   * @brief Creates a non-owning view of the entire array.
   *
   * @return An `NDView` that points to this array's data.
   */
  [[nodiscard]] inline NDView<T, S> view() const {
    if (ndim_ == 0) return NDView<T, S>();
    return NDView<T, S>(data_, shape_, strides_, ndim_);
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
    if (ndim_ > 0) {
      kakuhen::util::serialize::serialize_array<S>(out, shape_, static_cast<size_t>(ndim_));
      kakuhen::util::serialize::serialize_one<S>(out, total_size_);
      kakuhen::util::serialize::serialize_array<T>(out, data_, static_cast<size_t>(total_size_));
    }
  }

  /*!
   * @brief Deserializes the array's metadata and data from a stream.
   *
   * The array is left unchanged if reading or allocation fails.
   *
   * @param in The input stream to read from.
   * @param with_type If true, expects and verifies type information in the stream.
   * @throws std::runtime_error if the stream is truncated, type information mismatches,
   *         or the array size overflows.
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
    S ndim_in;
    kakuhen::util::serialize::deserialize_one<S>(in, ndim_in);
    if (!std::in_range<size_t>(ndim_in)) {
      throw std::runtime_error("NDArray: rank overflow during deserialization");
    }
    std::vector<S> shape_in(static_cast<size_t>(ndim_in));
    kakuhen::util::serialize::deserialize_array<S>(in, shape_in.data(), shape_in.size());
    load(in, shape_in);
  }

  /*!
   * @brief Deserializes an array whose shape must match `expected_shape`.
   *
   * Use this for untrusted streams: the stored shape is checked against
   * `expected_shape` before allocation, and the complete allocation size is
   * checked for overflow. `expected_shape` may refer to this array's own shape.
   * The array is left unchanged if reading or allocation fails.
   *
   * @param in The input stream to read from.
   * @param expected_shape The shape the stored array must have.
   * @throws std::runtime_error if the stream is truncated, the stored shape
   *         differs from `expected_shape`, or the allocation size overflows.
   */
  void deserialize_expected_shape(std::istream& in, std::span<const S> expected_shape) {
    using kakuhen::util::serialize::deserialize_one;
    S ndim_in;
    deserialize_one<S>(in, ndim_in);
    if (std::cmp_not_equal(ndim_in, expected_shape.size())) {
      throw std::runtime_error("NDArray: shape mismatch during deserialization");
    }
    for (const S extent : expected_shape) {
      S extent_in;
      deserialize_one<S>(in, extent_in);
      if (extent_in != extent) {
        throw std::runtime_error("NDArray: shape mismatch during deserialization");
      }
    }
    load(in, expected_shape);
  }

 private:
  S ndim_;
  S total_size_;

  std::unique_ptr<std::byte[]> memory_block_;
  S* shape_ = nullptr;
  S* strides_ = nullptr;
  T* data_ = nullptr;

  // Keep the old storage alive: `shape` may refer to it, and reading may throw.
  void load(std::istream& in, std::span<const S> shape) {
    NDArray loaded(shape);
    if (loaded.ndim_ != 0) {
      S total_size_in;
      kakuhen::util::serialize::deserialize_one<S>(in, total_size_in);
      if (total_size_in != loaded.total_size_) {
        throw std::runtime_error("NDArray: total size mismatch during deserialization");
      }
      kakuhen::util::serialize::deserialize_array<T>(in, loaded.data_,
                                                     static_cast<size_t>(loaded.total_size_));
    }
    *this = std::move(loaded);
  }

  // Used only by constructors; all size arithmetic is checked before allocation.
  // `S` counts elements, `size_t` counts bytes; this is where the two meet.
  void init(S ndim, const S* shape) {
    static_assert(alignof(T) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
                  "NDArray: over-aligned T is unsupported by the single-allocation strategy");
    constexpr size_t MAX_BYTES = std::numeric_limits<size_t>::max();
    // S may be wider than size_t, so clamp instead of converting.
    constexpr size_t MAX_S = std::in_range<size_t>(std::numeric_limits<S>::max())
                                 ? static_cast<size_t>(std::numeric_limits<S>::max())
                                 : std::numeric_limits<size_t>::max();
    if (!std::in_range<size_t>(ndim) || static_cast<size_t>(ndim) > MAX_BYTES / (2 * sizeof(S))) {
      throw std::runtime_error("NDArray: size overflow");
    }
    ndim_ = ndim;
    total_size_ = 0;
    if (ndim == 0) return;

    const size_t shape_bytes = static_cast<size_t>(ndim) * sizeof(S);
    const size_t metadata_bytes = 2 * shape_bytes;
    const size_t padding = (alignof(T) - metadata_bytes % alignof(T)) % alignof(T);
    if (padding > MAX_BYTES - metadata_bytes) {
      throw std::runtime_error("NDArray: size overflow");
    }
    const size_t data_offset = metadata_bytes + padding;
    size_t count = 1;
    // Reverse order checks intermediate strides too, even for shapes containing zero.
    for (S i = ndim; i-- > 0;) {
      if (!std::in_range<size_t>(shape[i]) ||
          (shape[i] != 0 && count > MAX_S / static_cast<size_t>(shape[i]))) {
        throw std::runtime_error("NDArray: size overflow");
      }
      count *= static_cast<size_t>(shape[i]);
    }
    if (count > (MAX_BYTES - data_offset) / sizeof(T)) {
      throw std::runtime_error("NDArray: size overflow");
    }
    total_size_ = static_cast<S>(count);
    const size_t total_bytes = data_offset + count * sizeof(T);

    // Allocate one block for metadata and elements.
    memory_block_.reset(new std::byte[total_bytes]);

    // Set up pointers.
    std::byte* base = memory_block_.get();
    shape_ = reinterpret_cast<S*>(base);
    strides_ = reinterpret_cast<S*>(base + shape_bytes);
    data_ = reinterpret_cast<T*>(base + data_offset);

    // Initialize metadata.
    std::copy_n(shape, ndim, shape_);
    compute_strides();

    // Initialize elements.
    if constexpr (!std::is_trivially_constructible_v<T>) {
      std::uninitialized_default_construct_n(data_, total_size_);
    }
  }

  void compute_strides() {
    S stride = 1;
    for (S i = ndim_; i-- > 0;) {
      strides_[i] = stride;
      stride *= shape_[i];
    }
  }

};  // class NDArray

template <typename T, typename S>
NDView<T, S>::NDView(NDArray<T, S>& arr) : NDView(arr.view()) {}

}  // namespace kakuhen::ndarray
