#pragma once

namespace kakuhen::histogram {

/*!
 * @brief A strongly-typed identifier for a registered axis or set of axes.
 *
 * Wraps an offset into the global AxisData storage and the number of dimensions.
 *
 * @tparam S The size type used for indexing (e.g., uint32_t).
 */
template <typename S>
class AxisId {
 public:
  using size_type = S;

  /**
   * @brief Constructs an AxisId.
   * @param offset The offset in the internal AxisData storage.
   * @param ndim The number of dimensions (default 1).
   */
  constexpr explicit AxisId(S offset, S ndim = 1) noexcept : id_(offset), ndim_(ndim) {}

  /**
   * @brief Get the offset in AxisData.
   * @return The raw offset value.
   */
  [[nodiscard]] constexpr S id() const noexcept {
    return id_;
  }

  /**
   * @brief Get the number of dimensions.
   * @return The dimension count.
   */
  [[nodiscard]] constexpr S ndim() const noexcept {
    return ndim_;
  }

  /**
   * @brief Conversion operator to the underlying size type (offset).
   * @return The raw offset value.
   */
  [[nodiscard]] constexpr operator S() const noexcept {
    return id_;
  }

  /**
   * @brief Default three-way comparison operator.
   */
  auto operator<=>(const AxisId&) const = default;

 private:
  S id_;    //!< The internal offset into the axis storage (AxisData).
  S ndim_;  //!< Number of dimensions.
};

}  // namespace kakuhen::histogram
