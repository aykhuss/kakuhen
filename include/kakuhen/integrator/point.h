#pragma once

#include "numeric_traits.h"
#include <vector>

//> implement the Monte Carlo context that is passed as an argument
//> to the integrand of the function/functor to be integrated

namespace kakuhen::integrator {

// template <typename T, typename S, typename U>
template <typename NT = num_traits_t<>>
// template <typename... Args>
struct Point {
  // using value_type = T;
  // using size_type = S;
  // using count_type = U;
  // using value_type = typename Traits::value_type;
  // using size_type = typename Traits::size_type;
  // using count_type = typename Traits::count_type;
  // using num_traits = typename num_traits<Args...>::type;
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  using count_type = typename num_traits::count_type;

  Point(size_type ndim, void* user_data = nullptr)
      : x(ndim, value_type(0)),
        weight(value_type(1)),
        ndim(ndim),
        sample_index(0),
        user_data(user_data) {}

  std::vector<value_type> x;
  value_type weight;
  size_type ndim;
  count_type sample_index;
  void* user_data;
};  // struct Point

// helper alias
template <typename... Args>
using point_t = Point<typename num_traits_of<Args...>::type>;

// num_traits_of specialization: A Point with some Traits
template <typename... Args>
struct num_traits_of<Point<Args...>> {
  using type = typename num_traits_of<Args...>::type;
};

}  // namespace kakuhen::integrator
