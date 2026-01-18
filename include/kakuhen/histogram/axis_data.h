#pragma once

#include <concepts>
#include <initializer_list>
#include <iterator>
#include <utility>
#include <vector>

namespace kakuhen::histogram {

/*!
 * @brief Storage for axis binning data.
 *
 * Stores continuous arrays of bin edges (or other data) for variable binning axes.
 * This improves data locality by keeping all edge definitions in a single
 * contiguous block of memory.
 *
 * @tparam T The coordinate value type (e.g., double).
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
   * @return The offset index where the appended data begins.
   */
  template <typename Range>
    requires requires(const Range& r) {
      std::begin(r);
      std::end(r);
    }
  [[nodiscard]] S add_data(const Range& range) {
    S offset = static_cast<S>(data_.size());
    auto first = std::begin(range);
    auto last = std::end(range);
    data_.insert(data_.end(), first, last);
    return offset;
  }

  /*!
   * @brief Appends data from an initializer list.
   *
   * Example: `axis_data.add_data({0.0, 1.0, 2.5, 5.0});`
   *
   * @param data The initializer list of values.
   * @return The offset index where the appended data begins.
   */
  [[nodiscard]] S add_data(std::initializer_list<T> data) {
    return add_data<std::initializer_list<T>>(data);
  }

  /*!
   * @brief Appends individual values to the storage (Variadic).
   *
   * Example: `axis_data.add_data(0.0, 1.0, 2.5, 5.0);`
   *
   * @tparam Args Variadic argument types (must be convertible to T).
   * @param args The values to append.
   * @return The offset index where the appended data begins.
   */
  template <typename... Args>
    requires(sizeof...(Args) > 0) && (std::convertible_to<Args, T> && ...)
  [[nodiscard]] S add_data(Args&&... args) {
    S offset = static_cast<S>(data_.size());
    data_.reserve(data_.size() + sizeof...(Args));
    (data_.emplace_back(std::forward<Args>(args)), ...);
    return offset;
  }

  /// @}

  /// @name Accessors
  /// @{

  /*!
   * @brief Access the global data vector.
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
  void clear() {
    data_.clear();
  }

  /*!
   * @brief Reserves memory for the underlying vector.
   * @param capacity The number of elements to reserve.
   */
  void reserve(S capacity) {
    data_.reserve(capacity);
  }

  /// @}

 private:
  std::vector<T> data_;
};

}  // namespace kakuhen::histogram
