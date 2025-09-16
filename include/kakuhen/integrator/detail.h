#pragma once

#include "numeric_traits.h"
#include <type_traits>

namespace kakuhen::integrator::detail {

// ----- helper: function traits for lambdas / functors -----
template <typename T>
struct function_traits;

// specialization: const-qualified operator()
template <typename C, typename R, typename Arg>
struct function_traits<R (C::*)(Arg) const> {
  using argument_type = Arg;
  using return_type = R;
};

// specialization: non-const operator()
template <typename C, typename R, typename Arg>
struct function_traits<R (C::*)(Arg)> {
  using argument_type = Arg;
  using return_type = R;
};

// ----- remove_cvref_t replacement for C++17 -----
template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// ----- Helper alias -----
template <typename F>
using num_traits_arg_t = num_traits_t<remove_cvref_t<
    typename function_traits<decltype(&F::operator())>::argument_type>>;

}  // namespace kakuhen::integrator::detail
