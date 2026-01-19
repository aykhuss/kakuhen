#pragma once

namespace kakuhen::histogram {

/*!
 * @brief A strongly-typed identifier for a registered axis.
 *
 * Wraps an index to the internal axis storage.
 *
 * @tparam S The size type used for indexing (e.g., uint32_t).
 */
template <typename S>
class AxisId {
 public:
  using size_type = S;

  /**
   * @brief Constructs an AxisId.
   * @param id The underlying integer index.
   */
  constexpr explicit AxisId(S id) noexcept : id_(id) {}

  /**
   * @brief Get the underlying integer index.
   * @return The raw index value.
   */
  [[nodiscard]] constexpr S id() const noexcept {
    return id_;
  }

  /**
   * @brief Conversion operator to the underlying size type.
   *
   * This allows the ID to be used directly as an index in arrays or vectors.
   * @return The raw index value.
   */
  [[nodiscard]] constexpr operator S() const noexcept {
    return id_;
  }

  /**
   * @brief Default three-way comparison operator.
   */
  auto operator<=>(const AxisId&) const = default;

 private:
  S id_;  //!< The internal index into the axis storage.
};

}  // namespace kakuhen::histogram
