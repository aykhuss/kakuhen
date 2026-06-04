#pragma once

#include "kakuhen/integrator/generator_base.h"
#include "kakuhen/integrator/vegas.h"
#include "kakuhen/util/math.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace kakuhen::integrator {

template <typename NT = util::num_traits_t<>,
          typename RNG = typename IntegratorDefaults<NT>::rng_type,
          typename DIST = typename IntegratorDefaults<NT>::dist_type>
class VegasGenerator : public Vegas<NT, RNG, DIST>,
                       public GeneratorBase<VegasGenerator<NT, RNG, DIST>, NT> {
 public:
  using IntBase = Vegas<NT, RNG, DIST>;
  using GenBase = GeneratorBase<VegasGenerator<NT, RNG, DIST>, NT>;
  using typename IntBase::cell_ctx_type;
  using typename IntBase::count_type;
  using typename IntBase::int_acc_type;
  using typename IntBase::point_type;
  using typename IntBase::size_type;
  using typename IntBase::value_type;

  // shorthands to save typing
  using S = typename IntBase::size_type;
  using T = typename IntBase::value_type;
  using U = typename IntBase::count_type;

  // load in Base members
  using IntBase::ndim_;
  using IntBase::ndiv_;
  using IntBase::opts_;
  using IntBase::ran;
  using IntBase::result_;

  /*!
   * @brief Construct a new VegasGenerator object.
   *
   * @param ndim The number of dimensions of the integration.
   * @param ndiv The number of divisions of the grid along each dimension.
   */
  explicit VegasGenerator(S ndim, S ndiv = 128)
      : IntBase(ndim, ndiv), gen_max_abs_val_({ndim, ndiv}) {};

  // /*!
  //  * @brief Construct a new VegasGenerator object by loading state from a file.
  //  *
  //  * @param filepath The path to the file containing the integrator state.
  //  */
  // explicit VegasGenerator(const std::filesystem::path& filepath) : IntBase(filepath) {
  //   IntBase::load(filepath);
  // }

  inline bool is_frozen() const {
    return opts_.frozen && *opts_.frozen;
  }

  template <typename I>
  int_acc_type optimize_bound_impl(I&& integrand, U neval) {
    /* todo: expose `npass` and `safety_factor` */

    gen_result_.reset();

    point_type point{ndim_, opts_.user_data.value_or(nullptr)};
    std::vector<T> u_buf(ndim_);
    cell_ctx_type cell({ndim_});

    // Pass 0: estimate the abs-integral A on uniform-u points and seed a flat,
    // strictly positive envelope with B == A.
    for (U i = 0; i < neval; ++i) {
      for (S idim = 0; idim < ndim_; ++idim)
        u_buf[idim] = IntBase::ran();
      point.sample_index = i;
      IntBase::map_point(u_buf, point, cell);
      const T fval = point.weight * integrand(point);
      const T abs_fval = util::math::abs(fval);
      gen_result_.accumulate(abs_fval, abs_fval * abs_fval);
    }
    bound_init_impl(gen_result_.value());

    // Pass 1: raise the envelope wherever it is exceeded.
    for (U i = 0; i < neval; ++i) {
      for (S idim = 0; idim < ndim_; ++idim)
        u_buf[idim] = IntBase::ran();
      point.sample_index = i;
      IntBase::map_point(u_buf, point, cell);
      const T fval = point.weight * integrand(point);
      const T abs_fval = util::math::abs(fval);
      if (abs_fval > bound_at(cell)) bound_raise(cell);
      gen_result_.accumulate(abs_fval, abs_fval * abs_fval);
    }

    return gen_result_;
  }

  template <typename I, typename ECB>
  int_acc_type generate_events_impl(I&& integrand, U neval, ECB&& event_call_back, T event_norm) {
    gen_result_.reset();

    point_type point{ndim_, opts_.user_data.value_or(nullptr)};
    T event_weight;
    std::vector<T> u_buf(ndim_);
    cell_ctx_type cell({ndim_}), cell_debug({ndim_});

    for (U i = 0; i < neval; ++i) {
      propose(u_buf, cell);
      point.sample_index = i;
      IntBase::map_point(u_buf, point, cell_debug);
      // assert cell == cell_debug
      const T fval = point.weight * integrand(point);
      const T abs_fval = util::math::abs(fval);
      // hit or miss
      const T abs_fval_bound = bound_at(cell);
      if (abs_fval > abs_fval_bound) {
        // overweight event...
        event_weight = kakuhen::util::math::sgn(fval) * (abs_fval / abs_fval_bound) * event_norm;
      } else {
        // rejection sampling
        const T r = ran();
        if (r < abs_fval / abs_fval_bound) {
          event_weight = T(kakuhen::util::math::sgn(fval)) * event_norm;
        } else {
          // event rejected, skip call-back and accumulation
          continue;
        }
      }
      event_call_back(point, event_weight);
      gen_result_.accumulate(abs_fval, abs_fval * abs_fval);
    }  // end for i

    return gen_result_;  // is this really the integral of |f|?
  }

  /// @name RNG hooks
  ///
  /// These touch the integrator's protected RNG (`ran()`), so they must live on
  /// the derived class; `GeneratorBase` only orchestrates and does pure math.
  /// @{

  /*!
   * @brief Draw a proposal point distributed proportional to B / V.
   *
   * For each dimension, the cell `ig` is chosen with probability proportional to
   * `gen_max_abs_val_(idim, ig)` via a linear CDF walk, then the in-cell
   * coordinate is drawn uniformly within the equal-width cell:
   * `u_idim = (ig + r) / ndiv`.
   */
  inline void propose(std::span<T> u, cell_ctx_type& ctx) {
    assert(u.size() == static_cast<std::size_t>(ndim_));
    assert(ctx.size() == static_cast<std::size_t>(ndim_));
    for (S idim = 0; idim < ndim_; ++idim) {
      // pick a cell ig with probability proportional to gen_max_abs_val_(idim, ig)
      T total = T(0);
      for (S ig = 0; ig < ndiv_; ++ig)
        total += gen_max_abs_val_(idim, ig);
      const T target = ran() * total;
      S ig = 0;
      T cum = gen_max_abs_val_(idim, 0);
      while (ig + 1 < ndiv_ && cum < target) {
        ++ig;
        cum += gen_max_abs_val_(idim, ig);
      }
      // uniform within the equal-width cell
      u[idim] = (T(ig) + ran()) / T(ndiv_);
      ctx[idim] = ig;
    }  // for idim
  }

  /// @}

  /// @name Bound-model hooks
  ///
  /// Pure operations on the separable envelope `gen_max_abs_val_(idim, ig)`; no RNG.
  /// @{

  /*!
   * @brief Seed a flat envelope whose product equals `abs_integral`.
   *
   * Sets every `gen_max_abs_val_(idim, ig) = abs_integral^(1/ndim)`, so
   * `B(u) == abs_integral` everywhere and the initial bound volume equals
   * `abs_integral`. A small floor guards against a degenerate zero/negative seed.
   */
  inline void bound_init_impl(T abs_integral) {
    const T eps = T(10) * std::numeric_limits<T>::min();
    const T base = util::math::max(abs_integral, eps);
    const T per = std::pow(base, T(1) / T(ndim_));
    gen_max_abs_val_.fill(per);
  }

  /// @brief Evaluate the separable bound at `ctx`: the product over dimensions
  ///        `B = prod_idim gen_max_abs_val_(idim, ctx[idim])`.
  [[nodiscard]] inline T bound_at(const cell_ctx_type& ctx) const {
    T b = T(1);
    for (S idim = 0; idim < ndim_; ++idim)
      b *= gen_max_abs_val_(idim, ctx[idim]);
    return b;
  }

  /*!
   * @brief Raise every cell touched by `ctx` (MINT rule).
   *
   * Multiplies each touched `gen_max_abs_val_(idim, cell_idim)` by
   * `1 + 1/(10*ndim)`; the product `B` therefore grows by
   * `(1 + 1/(10*ndim))^ndim` at the violating point.
   */
  inline void bound_raise(const cell_ctx_type& ctx) {
    const T fac = T(1) + T(1) / (T(10) * T(ndim_));
    for (S idim = 0; idim < ndim_; ++idim)
      gen_max_abs_val_(idim, ctx[idim]) *= fac;
  }

  /// @brief Scale the whole envelope by `factor` (e.g. a safety factor).
  inline void bound_scale(T factor) {
    const T per = std::pow(factor, T(1) / T(ndim_));
    for (auto& v : gen_max_abs_val_)
      v *= per;
  }

  [[nodiscard]] inline T bound_volume() const {
    T vol = T(1);
    for (S idim = 0; idim < ndim_; ++idim) {
      T sum = T(0);
      for (S idiv = 0; idiv < ndiv_; ++idiv)
        sum += gen_max_abs_val_(idim, idiv);
      vol *= sum / T(ndiv_);
    }
    return vol;
  }

  /// @}

 private:
  /// @brief Classify a uniform deviate into an equal-width cell along dimension `idim`.
  [[nodiscard]] inline S cell_of([[maybe_unused]] S idim, T r) const noexcept {
    const S nc = ndiv_;
    const S c = static_cast<S>(r * T(nc));
    return c < nc ? c : nc - 1;  // guard the r == 1 boundary
  }

  /// data structures needed for the generation of events
  ndarray::NDArray<T, S> gen_max_abs_val_;
  //@todo: keep track of the all-time-max value too?

  int_acc_type gen_result_;
};

}  // namespace kakuhen::integrator
