#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace kakuhen::util::math {

template <typename T>
[[nodiscard]] constexpr bool nearly_equal(const T& a, const T& b, std::size_t max_steps = 4,
                                          T abs_tol = std::numeric_limits<T>::epsilon()) {
  static_assert(std::is_floating_point_v<T>, "T must be floating-point");

  // Special cases
  if (std::isnan(a) || std::isnan(b)) return false;
  if (std::isinf(a) || std::isinf(b)) return a == b;
  if (a == b) return true;  // also handles +0.0 vs -0.0

  const T diff = std::abs(a - b);

  const T ulp_a = std::abs(std::nextafter(a, b) - a);
  const T ulp_b = std::abs(std::nextafter(b, a) - b);
  const T threshold = std::max(ulp_a, ulp_b) * max_steps;

  return diff <= std::max(threshold, abs_tol);
}

}  // namespace kakuhen::util::math
