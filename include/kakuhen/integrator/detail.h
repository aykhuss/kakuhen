#pragma once

#include "kakuhen/util/numeric_traits.h"
#include <type_traits>

namespace kakuhen::integrator::detail {

/// @name Function Traits
/// @{

/*!
 * @brief Helper struct to extract argument and return types from function objects.
 *
 * This primary template is undefined and will cause a compile-time error if
 * instantiated with a type that is not a pointer to member function.
 *
 * @tparam T The type to inspect.
 */
template <typename T>
struct function_traits;

/*!
 * @brief Specialization for const-qualified member functions (e.g., lambdas).
 *
 * @tparam C The class type.
 * @tparam R The return type.
 * @tparam Arg The argument type.
 */
template <typename C, typename R, typename Arg>
struct function_traits<R (C::*)(Arg) const> {
  using argument_type = Arg;
  using return_type = R;
};

/*!
 * @brief Specialization for non-const member functions.
 *
 * @tparam C The class type.
 * @tparam R The return type.
 * @tparam Arg The argument type.
 */
template <typename C, typename R, typename Arg>
struct function_traits<R (C::*)(Arg)> {
  using argument_type = Arg;
  using return_type = R;
};

/// @}

/// @name Traits Deduction
/// @{

/*!
 * @brief Helper alias to deduce NumericTraits from a functor's argument type.
 *
 * This alias inspects the `operator()` of the given functor type `F`, extracts
 * its argument type, removes const/volatile/reference qualifiers, and then
 * uses `num_traits_t` to determine the corresponding `NumericTraits`.
 *
 * This is used to automatically deduce numeric traits from the integrand
 * function passed to the integrator.
 *
 * @tparam F The functor type (e.g., lambda or struct with operator()).
 */
template <typename F>
using num_traits_arg_t = kakuhen::util::num_traits_t<
    std::remove_cvref_t<typename function_traits<decltype(&F::operator())>::argument_type>>;

/// @}

}  // namespace kakuhen::integrator::detail
