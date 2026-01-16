#pragma once

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

  friend constexpr bool operator==(const HistogramId& lhs, const HistogramId& rhs) noexcept {
    return lhs.id_ == rhs.id_;
  }

  friend constexpr bool operator!=(const HistogramId& lhs, const HistogramId& rhs) noexcept {
    return lhs.id_ != rhs.id_;
  }

 private:
  S id_;
};

}  // namespace kakuhen::histogram
