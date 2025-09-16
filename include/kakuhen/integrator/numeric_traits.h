#pragma once

#include <cstdint>

namespace kakuhen::integrator {

template <typename T, typename S, typename U>
struct NumericTraits {
  using value_type = T;
  using size_type = S;
  using count_type = U;
};

// ----- helper: extract NumericTraits -----

// Raw triplet (T,S,U) with defaults
template <typename T = double, typename S = std::uint32_t,
          typename U = std::uint64_t>
struct num_traits_of {
  using type = NumericTraits<T, S, U>;
};

// template <typename... Args>
// struct num_traits_of;

// specialization: No args -> default NumericTraits
// template <>
// struct num_traits_of<> {
//   using type = NumericTraits<>;
// };

// specialization: Already a NumericTraits
template <typename T, typename S, typename U>
struct num_traits_of<NumericTraits<T, S, U>> {
  using type = NumericTraits<T, S, U>;
};

// ----- Helper alias -----
template <typename... Args>
using num_traits_t = typename num_traits_of<Args...>::type;

}  // namespace kakuhen::integrator
