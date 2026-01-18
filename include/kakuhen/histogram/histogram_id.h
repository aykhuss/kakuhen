#pragma once

#include <compare>

namespace kakuhen::histogram {

/*!
 * @brief A strongly-typed identifier for a registered histogram.
 *
 * Wraps an index to the internal registry storage.
 *
 * @tparam S The size type used for indexing (e.g., uint32_t).
 */
template <typename S>
class HistogramId {
 public:
  using size_type = S;

  constexpr explicit HistogramId(S id) noexcept : id_(id) {}

  /*!
   * @brief Get the underlying integer index.
   */
  [[nodiscard]] constexpr S id() const noexcept { return id_; }

  /*!
   * @brief Conversion operator to the underlying size type.
   * Allows the ID to be used as an index directly.
   */
  constexpr operator S() const noexcept { return id_; }

  // Default comparison operators (C++20)
  auto operator<=>(const HistogramId&) const = default;

 private:
  S id_;
};

}  // namespace kakuhen::histogram