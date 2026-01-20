#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace kakuhen::util::math {

/*!
 * @brief Computes the absolute value of a number.
 *
 * This function provides a compile-time (constexpr) compatible absolute value
 * calculation, which is necessary because `std::abs` is not `constexpr` until C++23.
 *
 * @tparam T The numeric type.
 * @param x The value.
 * @return The absolute value of x.
 */
template <typename T>
[[nodiscard]] constexpr T abs(T x) {
  return x < T(0) ? -x : x;
}

/*!
 * @brief Computes the square of a number.
 *
 * A convenience function for `x * x`. This is often more efficient and readable
 * than `std::pow(x, 2)`.
 *
 * @tparam T The numeric type.
 * @param x The value to square.
 * @return The square of x.
 */
template <typename T>
[[nodiscard]] constexpr T sq(const T& x) noexcept {
  return x * x;
}

/*!
 * @brief Extracts the sign of a value.
 *
 * @tparam T The numeric type.
 * @param val The value to check.
 * @return -1 if val < 0, 1 if val > 0, 0 if val == 0.
 */
template <typename T>
[[nodiscard]] constexpr int sgn(const T& val) noexcept {
  return (T(0) < val) - (val < T(0));
}

/*!
 * @brief Returns the smaller of two values.
 *
 * Provided as a `constexpr` alternative to `std::min`.
 *
 * @tparam T The value type.
 * @param a First value.
 * @param b Second value.
 * @return The smaller of a and b.
 */
template <typename T>
[[nodiscard]] constexpr const T& min(const T& a, const T& b) noexcept {
  return (b < a) ? b : a;
}

/*!
 * @brief Returns the larger of two values.
 *
 * Provided as a `constexpr` alternative to `std::max`.
 *
 * @tparam T The value type.
 * @param a First value.
 * @param b Second value.
 * @return The larger of a and b.
 */
template <typename T>
[[nodiscard]] constexpr const T& max(const T& a, const T& b) noexcept {
  return (a < b) ? b : a;
}

/*!
 * @brief Computes base raised to the power of exp using binary exponentiation.
 *
 * This function uses the "exponentiation by squaring" algorithm, which achieves
 * the result in O(log exp) multiplications. This is generally much faster than
 * `std::pow` for integer exponents.
 *
 * @tparam T The base type.
 * @tparam Integer The exponent type (must be integral).
 * @param base The base value.
 * @param exp The integer exponent.
 * @return The result of base^exp.
 */
template <typename T, typename Integer>
[[nodiscard]] constexpr T ipow(T base, Integer exp) {
  static_assert(std::is_integral_v<Integer>, "Exponent must be an integer");

  if (exp < 0) {
    if constexpr (std::is_floating_point_v<T>) {
      return T(1) / ipow(base, -exp);
    } else {
      // For integer bases, x^-n is 0 unless x is 1 or -1
      return (base == 1) ? 1 : ((base == -1) ? (exp & 1 ? -1 : 1) : 0);
    }
  }

  T result = 1;
  while (exp > 0) {
    if (exp & 1) result *= base;
    base *= base;
    exp >>= 1;
  }
  return result;
}

/*!
 * @brief Checks if two floating-point numbers are nearly equal.
 *
 * This implementation uses a relative error comparison which is significantly
 * faster than `std::nextafter` and friendly to SIMD optimizations.
 *
 * The check performed is effectively:
 * \f$ |a - b| \le \max(\text{abs\_tol}, \max(|a|, |b|) \times \epsilon \times \text{max\_ulps}) \f$
 *
 * @tparam T The floating-point type (e.g., float, double).
 * @param a The first number to compare.
 * @param b The second number to compare.
 * @param max_ulps The maximum number of "epsilon steps" allowed. Defaults to 4.
 * @param abs_tol The absolute tolerance for comparisons near zero. Defaults to machine epsilon.
 * @return True if the numbers are nearly equal, false otherwise.
 */
template <typename T>
[[nodiscard]] constexpr bool nearly_equal(T a, T b, int max_ulps = 4,
                                          T abs_tol = std::numeric_limits<T>::epsilon()) {
  static_assert(std::is_floating_point_v<T>, "T must be floating-point");

  // Shortcut for exact equality (handles +0.0 == -0.0)
  if (a == b) return true;

  // Handle NaN and Infinity if not in a constant evaluation context
  if (!std::is_constant_evaluated()) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (std::isinf(a) || std::isinf(b)) return false;  // a == b checked above
  }

  const T diff = abs(a - b);

  // 1. Check absolute tolerance (essential for numbers near zero)
  if (diff <= abs_tol) return true;

  // 2. Check relative tolerance (Knuth's method)
  const T max_abs = max(abs(a), abs(b));
  return diff <= (max_abs * std::numeric_limits<T>::epsilon() * static_cast<T>(max_ulps));

  // // old implementation: not constexpr & slower
  // if (std::isnan(a) || std::isnan(b)) return false;
  // if (std::isinf(a) || std::isinf(b)) return a == b;
  // if (a == b) return true;  // also handles +0.0 vs -0.0
  // const T diff = std::abs(a - b);
  // const T ulp_a = std::abs(std::nextafter(a, b) - a);
  // const T ulp_b = std::abs(std::nextafter(b, a) - b);
  // const T threshold = std::max(ulp_a, ulp_b) * max_steps;
  // return diff <= std::max(threshold, abs_tol);
}

}  // namespace kakuhen::util::math
