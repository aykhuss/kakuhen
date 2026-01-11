#pragma once
#include <cstddef>
#include <optional>

namespace kakuhen::ndarray {

/// @brief Convenience nullopt alias, e.g., Slice(_, 5)
constexpr std::nullopt_t _ = std::nullopt;

/*!
 * @brief Defines a slice for multi-dimensional array access.
 *
 * This struct allows specifying start, stop, and step parameters for slicing
 * multi-dimensional arrays, similar to Python's slice objects. Optional values
 * allow for default behavior (e.g., slicing from beginning to end).
 *
 * @tparam S The type used for size and index values (e.g., `uint32_t`).
 */
template <typename S>
struct Slice {
  using size_type = S;

  std::optional<S> start;  //!< The starting index of the slice (inclusive).
  std::optional<S> stop;   //!< The ending index of the slice (exclusive).
  std::optional<S> step;   //!< The step size of the slice.

  /*!
   * @brief Constructs a slice representing a single element.
   *
   * Example: `Slice(3)` is equivalent to `[3:4]`.
   *
   * @param s The single index to slice.
   */
  constexpr Slice(S s) : start(s), stop(s + 1), step(1) {}

  /*!
   * @brief Constructs a full slice (equivalent to `[:]`).
   */
  constexpr Slice() : start(std::nullopt), stop(std::nullopt), step(std::nullopt) {}

  /*!
   * @brief Constructs an explicit slice with optional start, stop, and step.
   *
   * @param s The optional starting index.
   * @param e The optional ending index.
   * @param st The optional step size.
   */
  constexpr Slice(std::optional<S> s, std::optional<S> e, std::optional<S> st = std::nullopt)
      : start(s), stop(e), step(st) {}

  /*!
   * @brief Factory method to create a range slice.
   *
   * Example: `Slice::range(1, 5, 2)` for `[1:5:2]`.
   *
   * @param s The optional starting index.
   * @param e The optional ending index.
   * @param st The optional step size.
   * @return A `Slice` object configured for the specified range.
   */
  [[nodiscard]] static constexpr Slice range(std::optional<S> s, std::optional<S> e,
                                             std::optional<S> st = std::nullopt) {
    return Slice(s, e, st);
  }

  /*!
   * @brief Factory method to create a full slice (equivalent to `[:]`).
   *
   * @return A `Slice` object configured for a full slice.
   */
  [[nodiscard]] static constexpr Slice all() {
    return Slice();
  }
};  // struct Slice

}  // namespace kakuhen::ndarray
