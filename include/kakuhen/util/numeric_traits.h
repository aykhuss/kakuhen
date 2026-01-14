#pragma once

#include <cstdint>
#include <type_traits>

namespace kakuhen::util {

/*!
 * @brief Defines the numeric types used throughout the kakuhen classes.
 *
 * This struct serves as a central place to define the primary data types
 * for numerical calculations, sizing of dimensions, and counting of
 * evaluations. This allows for easy customization of precision and range
 * for different applications.
 *
 * @tparam T The value type for integral results (must be floating point).
 * @tparam S The size type for dimensions and array indices (must be integral).
 * @tparam U The count type for number of evaluations or samples (must be integral).
 */
template <typename T, typename S, typename U>
struct NumericTraits {
  static_assert(std::is_floating_point_v<T>, "Value type must be floating point");
  static_assert(std::is_integral_v<S>, "Size type must be integral");
  static_assert(std::is_integral_v<U>, "Count type must be integral");

  using value_type = T;  //!< The primary type for numerical values (e.g., integrand results).
  using size_type = S;   //!< The type for representing sizes, dimensions, and indices.
  using count_type = U;  //!< The type for counting samples, iterations, or evaluations.
};

/// @name Traits Extraction Helpers
/// @{

/*!
 * @brief Helper to extract or define NumericTraits from template arguments.
 *
 * This primary template defines `type` as `NumericTraits<T, S, U>`
 * with default types if not provided.
 *
 * @tparam T The value type. Defaults to `double`.
 * @tparam S The size type. Defaults to `std::uint32_t`.
 * @tparam U The count type. Defaults to `std::uint64_t`.
 */
template <typename T = double, typename S = std::uint32_t, typename U = std::uint64_t>
struct num_traits_of {
  using type = NumericTraits<T, S, U>;
};

/*!
 * @brief Specialization for when NumericTraits is already provided.
 *
 * If the template argument is already a `NumericTraits` type, this
 * specialization simply uses that type.
 *
 * @tparam T The value type.
 * @tparam S The size type.
 * @tparam U The count type.
 */
template <typename T, typename S, typename U>
struct num_traits_of<NumericTraits<T, S, U>> {
  using type = NumericTraits<T, S, U>;
};

/// @}

/// @name Aliases
/// @{

/*!
 * @brief Convenience alias for `typename num_traits_of<Args...>::type`.
 *
 * This alias simplifies the usage of `num_traits_of`, allowing users to
 * specify numeric traits more concisely.
 *
 * @tparam Args Template arguments to pass to `num_traits_of`.
 */
template <typename... Args>
using num_traits_t = typename num_traits_of<Args...>::type;

/// @}

}  // namespace kakuhen::util
