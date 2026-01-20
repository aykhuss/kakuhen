#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>

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

}  // namespace kakuhen::ndarray::detail
