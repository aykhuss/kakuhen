#pragma once

#include "kakuhen/integrator/integrator_base.h"
#include <cassert>
#include <cstddef>

namespace kakuhen::integrator {

/*!
 * @brief A naive Monte Carlo integrator.
 *
 * This class implements a simple (naive) Monte Carlo integration algorithm.
 * It samples points uniformly within the unit hypercube and accumulates the
 * function values. This integrator is straightforward but less efficient than
 * adaptive methods for many integrands, as it does not attempt to concentrate
 * samples in important regions. It does not support any special features like
 * adaptation or state serialization.
 *
 * @tparam NT The numeric traits for the integrator.
 * @tparam RNG The random number generator to use.
 * @tparam DIST The random number distribution to use.
 */
template <typename NT = util::num_traits_t<>,
          typename RNG = typename IntegratorDefaults<NT>::rng_type,
          typename DIST = typename IntegratorDefaults<NT>::dist_type>
class Plain : public IntegratorBase<Plain<NT, RNG, DIST>, NT, RNG, DIST> {
 public:
  static constexpr IntegratorId static_id() noexcept {
    return IntegratorId::PLAIN;
  }

  // dependent class: need to explicitly load things from the Base
  using Base = IntegratorBase<Plain<NT, RNG, DIST>, NT, RNG, DIST>;
  using typename Base::count_type;
  using typename Base::int_acc_type;
  using typename Base::num_traits;
  using typename Base::point_type;
  using typename Base::seed_type;
  using typename Base::size_type;
  using typename Base::value_type;

  //  member variables
  using Base::ndim_;
  using Base::opts_;

  /*!
   * @brief Construct a new Plain object.
   *
   * @param ndim The number of dimensions of the integration.
   */
  explicit Plain(size_type ndim) : Base(ndim) {
    assert(ndim > 0);
  };

  /// @name Integration Implementation
  /// @{

  /*!
   * @brief Implementation of the integration loop for a single iteration.
   *
   * @tparam I The type of the integrand function.
   * @param integrand The function to integrate.
   * @param neval The number of evaluations to perform.
   * @return An `int_acc_type` containing the accumulated results for this iteration.
   */
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

  /// @}

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
