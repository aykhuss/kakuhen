// Plain - a naive integrator
#pragma once

#include "kakuhen/integrator/integrator_base.h"
#include "kakuhen/integrator/numeric_traits.h"
#include "kakuhen/integrator/point.h"
#include <cassert>
#include <cstddef>

namespace kakuhen::integrator {

template <typename NT = num_traits_t<>, typename RNG = typename IntegratorDefaults<NT>::rng_type,
          typename DIST = typename IntegratorDefaults<NT>::dist_type>
class Plain : public IntegratorBase<Plain<NT, RNG, DIST>, NT, RNG, DIST> {
 public:
  static constexpr IntegratorId id = IntegratorId::PLAIN;
  static constexpr IntegratorFeature features = IntegratorFeature::NONE;

  // dependent class: need to explicitly load things from the Base
  using Base = IntegratorBase<Plain<NT, RNG, DIST>, NT, RNG, DIST>;
  using typename Base::count_type;
  using typename Base::int_acc_type;
  using typename Base::num_traits;
  using typename Base::point_type;
  using typename Base::seed_type;
  using typename Base::size_type;
  using typename Base::value_type;
  // using typename Base::result_type;
  //  member variables
  using Base::ndim_;
  using Base::opts_;

  explicit Plain(size_type ndim) : Base(ndim) {
    assert(ndim > 0);
  };

  template <typename I>
  int_acc_type integrate_impl(I&& integrand, count_type neval) {
    result_.reset();

    Point<num_traits> point{ndim_, opts_.user_data.value_or(nullptr)};

    for (count_type i = 0; i < neval; ++i) {
      generate_point(point, i);
      const value_type func = point.weight * integrand(point);
      const value_type func2 = func * func;
      result_.accumulate(func, func2);
    }

    return result_;
  }

 private:
  int_acc_type result_;

  inline void generate_point(Point<num_traits>& point, count_type sample_index = count_type(0)) {
    point.sample_index = sample_index;
    point.weight = value_type(1);
    for (size_type idim = 0; idim < ndim_; ++idim) {
      point.x[idim] = Base::ran();
    }
  }

};  // class Plain

}  // namespace kakuhen::integrator
