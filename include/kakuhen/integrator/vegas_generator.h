#pragma once

#include "kakuhen/integrator/generator_base.h"
#include "kakuhen/integrator/vegas.h"
#include "kakuhen/ndarray/ndarray.h"
#include <algorithm>
#include <span>
#include <vector>

namespace kakuhen::integrator {

/*!
 * @brief VEGAS integrator with event generation.
 *
 * `GeneratorBase` takes care of the generation itself and this class only
 * defines the envelope. It follows the sampling structure of VEGAS: one table
 * per dimension with `ndiv` equal-width cells in u-space (the same cells that
 * `map_point` picks), so that
 * `R(u) = prod_i envelope_(i, floor(u_i * ndiv))`. The `{ndim, ndiv}` table
 * itself is stored in `GeneratorBase`.
 *
 * @tparam NT The numeric traits for the integrator.
 * @tparam RNG The random number generator to use.
 * @tparam DIST The random number distribution to use.
 */
template <typename NT = util::num_traits_t<>,
          typename RNG = typename IntegratorDefaults<NT>::rng_type,
          typename DIST = typename IntegratorDefaults<NT>::dist_type>
class VegasGenerator : public GeneratorBase<VegasGenerator<NT, RNG, DIST>, Vegas<NT, RNG, DIST>> {
 public:
  using GenBase = GeneratorBase<VegasGenerator<NT, RNG, DIST>, Vegas<NT, RNG, DIST>>;
  using typename GenBase::IntBase;
  // GeneratorBase needs access to the envelope hooks below
  friend GenBase;

  using typename GenBase::point_type;
  using typename IntBase::cell_ctx_type;

  // shorthands to save typing
  using S = typename GenBase::size_type;
  using T = typename GenBase::value_type;

 protected:
  // load in Base members
  using GenBase::envelope_;
  using IntBase::grid_;
  using IntBase::ndim_;
  using IntBase::ndiv_;

 public:
  /*!
   * @brief Construct a new VegasGenerator object.
   *
   * @param ndim The number of dimensions of the integration.
   * @param ndiv The number of divisions of the grid along each dimension.
   */
  explicit VegasGenerator(S ndim, S ndiv = 128) : GenBase(ndim, ndiv) {}

 protected:
  /// @name Envelope hooks used by `GeneratorBase`
  /// @{

  /// @brief Cell-context buffer for the `map_point` of this integrator.
  [[nodiscard]] inline cell_ctx_type make_cell_ctx() const {
    return cell_ctx_type({ndim_});
  }

  /// @brief One row of `ndiv` cell factors per dimension.
  [[nodiscard]] std::vector<S> env_shape() const {
    return {ndim_, ndiv_};
  }

  /// @brief Evaluate `R = prod_i envelope_(i, cell_i)` using the cells
  ///        recorded by `map_point`.
  [[nodiscard]] inline T env_value(const cell_ctx_type& cell) const {
    assert(cell.size() == static_cast<std::size_t>(ndim_));
    T envelope = T(1);
    for (S idim = 0; idim < ndim_; ++idim)
      envelope *= envelope_(idim, cell[idim]);
    return envelope;
  }

  /// @brief Multiply the cells recorded in `cell` (one per dimension) by `gamma`.
  inline void env_raise(const cell_ctx_type& cell, T gamma) {
    for (S idim = 0; idim < ndim_; ++idim)
      envelope_(idim, cell[idim]) *= gamma;
  }

  /*!
   * @brief Draw a point with density R / V and map it through the grid.
   *
   * Uses one random number per dimension (inverse CDF). The cell `ig` is picked
   * with probability proportional to `envelope_(idim, ig)` by a binary search
   * in the inclusive prefix sums `envelope_cdf_`. The position of the random
   * number inside the CDF segment of that cell is uniform and independent of
   * the cell choice, so we re-use it as the position inside the cell. The
   * grid map of VEGAS is linear inside a cell, so that position goes straight
   * to x-space without a detour through u. `point.x` and `point.weight` are
   * set as `map_point` would set them for the drawn cells.
   *
   * @param point Output point whose coordinates and weight are overwritten.
   * @return The envelope value `R(u)` at the proposed point.
   */
  inline T env_propose(point_type& point) {
    return propose_impl<false>(point, nullptr);
  }

  /// @brief As `env_propose(point)`, and also record the drawn cells in `cell`
  ///        (as `map_point` would), such that `env_value(cell)` gives back R.
  inline T env_propose(point_type& point, cell_ctx_type& cell) {
    return propose_impl<true>(point, &cell);
  }

  /// @brief Rebuild the inclusive prefix sums of `envelope_` for each dimension,
  ///        which `env_propose` uses to find a cell in O(log ndiv).
  /// @return The volume `V = prod_i (sum_ig envelope_(i, ig)) / ndiv`.
  inline T env_prepare() {
    if (!std::ranges::equal(envelope_cdf_.shape(), envelope_.shape())) {
      envelope_cdf_ = ndarray::NDArray<T, S>(envelope_.shape());
    }
    T volume = T(1);
    for (S idim = 0; idim < ndim_; ++idim) {
      T sum = T(0);
      for (S ig = 0; ig < ndiv_; ++ig) {
        sum = detail::cdf_step(sum, envelope_(idim, ig));
        envelope_cdf_(idim, ig) = sum;
      }
      volume *= sum / T(ndiv_);
    }
    return volume;
  }

  /// @}

 private:
  template <bool RecordCells>
  inline T propose_impl(point_type& point, [[maybe_unused]] cell_ctx_type* cell) {
    assert(point.x.size() == static_cast<std::size_t>(ndim_));
    T envelope = T(1);
    T weight = T(1);
    for (S idim = 0; idim < ndim_; ++idim) {
      const auto draw = detail::sample_cdf<T, S>({&envelope_cdf_(idim, 0), ndiv_}, IntBase::ran());
      const S ig = draw.cell;
      const T x_low = ig > 0 ? grid_(idim, ig - 1) : T(0);
      const T x_upp = grid_(idim, ig);
      point.x[idim] = x_low + draw.fraction * (x_upp - x_low);
      weight *= T(ndiv_) * (x_upp - x_low);
      envelope *= draw.width;
      if constexpr (RecordCells) (*cell)[idim] = ig;
    }  // for idim
    point.weight = weight;
    return envelope;
  }

  /// inclusive prefix sums of `envelope_` for each dimension; rebuilt in
  /// `env_prepare` whenever the envelope is sealed and used in `env_propose`
  ndarray::NDArray<T, S> envelope_cdf_;
};

}  // namespace kakuhen::integrator
