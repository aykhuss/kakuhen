#pragma once

#include "numeric_traits.h"
#include <vector>

namespace kakuhen::integrator {

/*!
 * @brief Represents a sample point in the integration space.
 *
 * This struct encapsulates all the necessary information for a single sample
 * point during a Monte Carlo integration. It is passed to the integrand
 * function/functor.
 *
 * It owns the memory for the coordinates (`x`) using a `std::vector`.
 *
 * @tparam NT The numeric traits for the integrator, defining value_type,
 * size_type, and count_type.
 */
template <typename NT = num_traits_t<>>
struct Point {
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  using count_type = typename num_traits::count_type;

  /*!
   * @brief Constructs a Point with a specified dimensionality.
   *
   * @param dimensions The number of dimensions for the point.
   * @param user_data Optional pointer to user-defined data.
   */
  explicit Point(size_type dimensions, void* user_data = nullptr)
      : x(dimensions, value_type(0)),
        weight(value_type(1)),
        ndim(dimensions),
        sample_index(0),
        user_data(user_data) {}

  std::vector<value_type> x;  //!< The coordinates of the point in the integration space.
  value_type weight;          //!< The weight associated with this point (e.g. importance sampling).
  size_type ndim;             //!< The dimensionality of the integration space (matches x.size()).
  count_type sample_index;    //!< The index of the current sample in the sequence.
  void* user_data;            //!< Pointer to user-defined data passed to the integrand.
};  // struct Point

/// @name Aliases & Traits
/// @{

/*!
 * @brief Helper alias to create a Point type from variadic arguments.
 *
 * Example: `point_t<double>` creates `Point<NumericTraits<double, ...>>`.
 *
 * @tparam Args Arguments passed to `num_traits_of`.
 */
template <typename... Args>
using point_t = Point<typename num_traits_of<Args...>::type>;

/*!
 * @brief num_traits_of specialization for Point types.
 *
 * Allows extracting NumericTraits from a Point type directly.
 *
 * @tparam Args Template arguments of the Point (which is expected to be a NumericTraits).
 */
template <typename... Args>
struct num_traits_of<Point<Args...>> {
  using type = typename num_traits_of<Args...>::type;
};

/// @}

}  // namespace kakuhen::integrator
