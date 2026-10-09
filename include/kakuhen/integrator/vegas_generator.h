#pragma once

#include "kakuhen/integrator/generator_base.h"
#include "kakuhen/integrator/vegas.h"
#include "kakuhen/ndarray/ndarray.h"
#include <span>
#include <utility>
#include <vector>

namespace kakuhen::integrator {

/*!
 * @brief VEGAS integrator with event generation
 *
 * `GeneratorBase` does the generation itself; this class only defines the
 * envelope. It follows the sampling structure of VEGAS: one table per
 * dimension with `ndiv` equal-width cells in u-space (the same cells
 * `map_point` picks), so R(u) = prod_i envelope_(i, floor(u_i * ndiv)).
 * The `{ndim, ndiv}` table itself lives in `GeneratorBase`.
 *
 * @tparam NT numeric traits for the integrator
 * @tparam RNG random number generator
 * @tparam DIST random number distribution
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
  using IntBase::grid_;
  using IntBase::ndim_;
  using IntBase::ndiv_;

 public:
  /*!
   * @brief construct a new VegasGenerator
   *
   * @param ndim number of dimensions
   * @param ndiv number of grid divisions along each dimension
   */
  explicit VegasGenerator(S ndim, S ndiv = 128) : GenBase(ndim, ndiv) {}

 protected:
  /// @name Envelope hooks used by `GeneratorBase`
  /// @{

  /// @brief cell-context buffer for `map_point`
  [[nodiscard]] inline cell_ctx_type make_cell_ctx() const {
    return cell_ctx_type({ndim_});
  }

  /// @brief one row of `ndiv` cell factors per dimension
  [[nodiscard]] std::vector<S> env_shape() const {
    return {ndim_, ndiv_};
  }

  /// @brief flat indices of the raw factors `envelope_(i, cell_i)`, one per dimension
  void env_indices(const cell_ctx_type& cell, std::span<S> indices) const {
    for (S idim = 0; idim < ndim_; ++idim)
      indices[idim] = idim * ndiv_ + cell[idim];
  }

  /*!
   * @brief draw a point with density R/V and map it through the grid
   *
   * One random number per dimension (inverse CDF). The cell `ig` is picked
   * with probability proportional to `envelope_(idim, ig)` by a binary search
   * in the inclusive prefix sums `envelope_cdf_`. Where the random number
   * lands inside the CDF segment of that cell is uniform and independent of
   * the cell choice, so we re-use it as the position inside the cell. VEGAS
   * maps linearly within a cell, so that position goes straight to x-space
   * without a detour through u. `point.x` and `point.weight` are set as
   * `map_point` would for the drawn cells.
   *
   * @param point output point; coordinates and weight are overwritten
   * @return envelope value R(u) at the proposed point
   */
  inline T env_propose(point_type& point) {
    return propose_impl<false>(point, nullptr);
  }

  /// @brief like `env_propose(point)`, but also record the drawn cells in `cell`
  ///        (as `map_point` would) so the factors at `env_indices(cell)` multiply to R
  inline T env_propose(point_type& point, cell_ctx_type& cell) {
    return propose_impl<true>(point, &cell);
  }

  /// @brief inclusive prefix sums of `table` per dimension; `env_propose` uses
  ///        them to find a cell in O(log ndiv)
  /// @return the volume V = prod_i (sum_ig table(i, ig)) / ndiv and the sums
  auto env_prepare(const ndarray::NDArray<T, S>& table) const {
    ndarray::NDArray<T, S> cdf(env_shape());
    T volume = T(1);
    for (S idim = 0; idim < ndim_; ++idim) {
      T sum = T(0);
      for (S ig = 0; ig < ndiv_; ++ig) {
        sum = detail::cdf_step(sum, table(idim, ig));
        cdf(idim, ig) = sum;
      }
      volume *= sum / T(ndiv_);
    }
    return std::pair{volume, std::move(cdf)};
  }

  void env_commit_cache(ndarray::NDArray<T, S>&& cdf) noexcept {
    envelope_cdf_ = std::move(cdf);
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

  /// inclusive prefix sums of `envelope_` per dimension; rebuilt whenever the
  /// envelope is sealed, used in `env_propose`
  ndarray::NDArray<T, S> envelope_cdf_;
};

}  // namespace kakuhen::integrator
