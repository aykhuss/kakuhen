#pragma once

#include "kakuhen/integrator/integrator_base.h"
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace kakuhen::integrator {

/*!
 * @brief A naive Monte Carlo integrator.
 *
 * This class implements a simple (naive) Monte Carlo integration algorithm.
 * It samples points uniformly within the unit hypercube and accumulates the
 * function values. This integrator is straightforward but less efficient than
 * adaptive methods for many integrands, as it does not attempt to concentrate
 * samples in important regions. It does not support adaptive grids or
 * integrator-specific state serialization.
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
  using typename Base::ProgressTracker;
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
   * @tparam ProgressCb The type of the progress callback (or std::nullptr_t).
   * @param integrand The function to integrate.
   * @param neval The number of evaluations to perform.
   * @param tracker Per-call progress bookkeeping shared with the base class.
   * @param progress_cb The progress callback for milestone notifications.
   * @return An `int_acc_type` containing the accumulated results for this iteration.
   */
  template <typename I, typename ProgressCb = std::nullptr_t>
  int_acc_type integrate_impl(I&& integrand, count_type neval,
                              [[maybe_unused]] ProgressTracker& tracker,
                              [[maybe_unused]] ProgressCb&& progress_cb = nullptr) {
    result_.reset();

    Point<num_traits> point{ndim_, opts_.user_data.value_or(nullptr)};
    std::vector<value_type> u_buf(ndim_);

    for (count_type i = 0; i < neval; ++i) {
      for (size_type idim = 0; idim < ndim_; ++idim)
        u_buf[idim] = Base::ran();
      point.sample_index = i;
      map_point_impl(u_buf, point);
      const value_type func = point.weight * integrand(point);
      const value_type func2 = func * func;
      result_.accumulate(func, func2);

      if constexpr (is_progress_callback_v<ProgressCb>) {
        if (Base::check_eval_milestone(tracker, progress_cb, i)) break;
      }
    }

    return result_;
  }

  /*!
   * @brief Maps caller-supplied uniform coordinates to a plain sample point.
   *
   * The plain integrator uses the identity map on the unit hypercube, so each
   * coordinate is copied directly from the corresponding entry in `u`. This
   * method does not draw from the RNG, does not mutate integrator state, and
   * leaves `point.sample_index` unchanged.
   *
   * @param u Uniform randoms in [0, 1), one per physical dimension.
   * @param point Output point whose coordinate buffer is filled from `u`; its
   *              weight is set to 1.
   */
  inline void map_point_impl(std::span<const value_type> u, Point<num_traits>& point) const {
    point.weight = value_type(1);
    for (size_type idim = 0; idim < ndim_; ++idim) {
      point.x[idim] = u[idim];
    }
  }

  /// @}

 private:
  int_acc_type result_;

  /*!
   * @brief Generates one uniformly distributed sample point in the unit hypercube.
   *
   * @param point The point object to populate.
   * @param sample_index Zero-based sample index stored in the point metadata.
   */
  inline void generate_point(Point<num_traits>& point, count_type sample_index = count_type(0)) {
    point.sample_index = sample_index;
    point.weight = value_type(1);
    for (size_type idim = 0; idim < ndim_; ++idim) {
      point.x[idim] = Base::ran();
    }
  }

};  // class Plain

}  // namespace kakuhen::integrator
