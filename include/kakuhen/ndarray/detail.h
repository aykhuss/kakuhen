#pragma once

#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace kakuhen::ndarray::detail {

/*!
 * @brief Computes the flat index from multi-dimensional indices.
 *
 * This function calculates the linear offset for a multi-dimensional array
 * given a set of indices, strides, and the array shape.
 *
 * @tparam S The type used for size and indices.
 * @param idx A pointer to the array of indices.
 * @param strides A pointer to the array of strides.
 * @param shape A pointer to the array of dimension sizes.
 * @param ndim The number of dimensions.
 * @return The calculated flat index.
 */
template <typename S>
[[nodiscard]] inline S flat_index(const S* idx, const S* strides, [[maybe_unused]] const S* shape,
                                  S ndim) noexcept {
  S offset = 0;
  for (S i = 0; i < ndim; ++i) {
    if constexpr (std::is_signed_v<S>) {
      assert(idx[i] >= 0);
    }
    assert(idx[i] < shape[i]);
    offset += idx[i] * strides[i];
  }
  return offset;
}

/*!
 * @brief Computes the flat index from a fixed number of indices (unrolled).
 *
 * This helper avoids loops by unrolling the index computation at compile time.
 * Bounds checking is intentionally omitted for performance in release builds.
 *
 * @tparam S The type used for size and indices.
 * @tparam Tuple A tuple holding the indices.
 * @tparam I Index sequence for the tuple elements.
 * @param idx A tuple containing the indices.
 * @param strides A pointer to the array of strides.
 * @return The calculated flat index.
 */
template <typename S, typename Tuple, std::size_t... I>
[[nodiscard]] inline S flat_index_unrolled(const Tuple& idx, const S* strides,
                                           std::index_sequence<I...>) noexcept {
  return (S(0) + ... + (static_cast<S>(std::get<I>(idx)) * strides[I]));
}

}  // namespace kakuhen::ndarray::detail
