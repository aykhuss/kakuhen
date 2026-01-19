#pragma once

#include <concepts>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kakuhen::histogram {

/*!
 * @brief Centralized storage for axis binning parameters and edges.
 *
 * Stores continuous arrays of bin edges (for variable axes) or parameters
 * (for uniform axes) in a single contiguous block of memory. This design
 * improves data locality and simplifies serialization of axis definitions.
 *
 * @tparam T The coordinate and parameter value type (e.g., double).
 * @tparam S The size/index type (e.g., uint32_t).
 */
template <typename T, typename S>
class AxisData {
 public:
  using value_type = T;
  using size_type = S;

  /// @name Adding Data
  /// @{

  /*!
   * @brief Appends a range of data to the storage.
   *
   * Supports any type that satisfies the range requirement (has std::begin/std::end),
   * such as std::vector, std::span, std::array, etc.
   *
   * @tparam Range The range type.
   * @param range The range of values to append.
   * @return The starting offset index of the appended data.
   * @throws std::length_error If the new size exceeds the capacity of the index type S.
   */
  template <typename Range>
    requires requires(const Range& r) {
      std::begin(r);
      std::end(r);
    }
  [[nodiscard]] S add_data(const Range& range) {
    const std::size_t offset = data_.size();
    const auto first = std::begin(range);
    const auto last = std::end(range);
    const std::size_t count = static_cast<std::size_t>(std::distance(first, last));

    if (offset + count > static_cast<std::size_t>(std::numeric_limits<S>::max())) {
      throw std::length_error("AxisData size exceeds capacity of index type S");
    }

    data_.insert(data_.end(), first, last);
    return static_cast<S>(offset);
  }

  /*!
   * @brief Appends data from an initializer list.
   *
   * @param data The initializer list of values.
   * @return The starting offset index of the appended data.
   */
  [[nodiscard]] S add_data(std::initializer_list<T> data) {
    return add_data<std::initializer_list<T>>(data);
  }

  /*!
   * @brief Appends individual values to the storage (Variadic).
   *
   * @tparam Args Variadic argument types (must be convertible to T).
   * @param args The values to append.
   * @return The starting offset index of the appended data.
   * @throws std::length_error If the new size exceeds the capacity of the index type S.
   */
  template <typename... Args>
    requires(sizeof...(Args) > 0) && (std::convertible_to<Args, T> && ...)
  [[nodiscard]] S add_data(Args&&... args) {
    const std::size_t offset = data_.size();
    if (offset + sizeof...(Args) > static_cast<std::size_t>(std::numeric_limits<S>::max())) {
      throw std::length_error("AxisData size exceeds capacity of index type S");
    }

    data_.reserve(offset + sizeof...(Args));
    (data_.emplace_back(std::forward<Args>(args)), ...);
    return static_cast<S>(offset);
  }

  /// @}

  /// @name Accessors
  /// @{

  /*!
   * @brief Access the global data vector.
   * @return A const reference to the underlying storage vector.
   */
  [[nodiscard]] const std::vector<T>& data() const noexcept {
    return data_;
  }

  /*!
   * @brief Access data element at index.
   *
   * @param index The index of the element to access.
   * @return A const reference to the element.
   */
  [[nodiscard]] const T& operator[](S index) const noexcept {
    return data_[static_cast<std::size_t>(index)];
  }

  /*!
   * @brief Access data element at index with bounds checking.
   *
   * @param index The index of the element to access.
   * @return A const reference to the element.
   * @throws std::out_of_range If index is invalid.
   */
  [[nodiscard]] const T& at(S index) const {
    return data_.at(static_cast<std::size_t>(index));
  }

  /*!
   * @brief Get the total number of stored elements.
   * @return The number of elements currently in storage.
   */
  [[nodiscard]] S size() const noexcept {
    return static_cast<S>(data_.size());
  }

  /// @}

  /// @name Management
  /// @{

  /*!
   * @brief Clears the storage.
   */
  void clear() noexcept {
    data_.clear();
  }

  /*!
   * @brief Reserves memory for the underlying vector to prevent reallocations.
   * @param capacity The number of elements to reserve.
   */
  void reserve(S capacity) {
    data_.reserve(static_cast<std::size_t>(capacity));
  }

  /// @}

 private:
  std::vector<T> data_;  //!< Contiguous storage for all axis data.
};

}  // namespace kakuhen::histogram
