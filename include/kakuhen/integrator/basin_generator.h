#pragma once

#include "kakuhen/integrator/basin.h"
#include "kakuhen/integrator/generator_base.h"
#include "kakuhen/ndarray/ndarray.h"
#include <algorithm>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace kakuhen::integrator {

/*!
 * @brief BASIN integrator with event generation
 *
 * `GeneratorBase` does the generation itself; this class only defines the
 * envelope. It follows the frozen BASIN sampling structure:
 *
 * - each diagonal dimension (entries `iord < nblocks` of the sampling order)
 *   has a 1D table over its `ndiv0` u-cells, the same cells `map_point` picks
 * - each conditional dimension has a table over `(ig1, ig2)`, with `ig1` the
 *   coarse cell of the parent dimension and `ig2` the conditional u-cell
 *   (same nesting as the `ordered_grid_` the map samples from)
 *
 * Since `ndiv0 = ndiv1 * ndiv2`, both kinds of tables fit into a single
 * `{ndim, ndiv1, ndiv2}` array indexed by the order entry. A diagonal row is
 * read flat as `ndiv0` cells, just like `ordered_grid_`/`ordered_grid0_` in
 * `Basin`.
 *
 * The envelope R is always the raw product of table entries. Sealing the
 * envelope computes the subtree integrals bottom-up, from the children to
 * their parents. Conditional cells are split at the coarse-grid edges in
 * x-space, so the product of the child integrals is constant on each
 * interval. The u-space masses of these intervals are exact and give the
 * cached CDFs, so no maximization or integration is needed during
 * generation. A proposal walks through the sampling order to sample
 * q = R / int R du with one CDF search per dimension and returns the raw
 * product R, the same one the raising passes use.
 *
 * @tparam NT numeric traits for the integrator
 * @tparam RNG random number generator
 * @tparam DIST random number distribution
 */
template <typename NT = util::num_traits_t<>,
          typename RNG = typename IntegratorDefaults<NT>::rng_type,
          typename DIST = typename IntegratorDefaults<NT>::dist_type>
class BasinGenerator : public GeneratorBase<BasinGenerator<NT, RNG, DIST>, Basin<NT, RNG, DIST>> {
 public:
  using GenBase = GeneratorBase<BasinGenerator<NT, RNG, DIST>, Basin<NT, RNG, DIST>>;
  using typename GenBase::IntBase;
  // GeneratorBase needs access to the envelope hooks below
  friend GenBase;

  /// the cell context type used by `Basin::map_point`
  using typename GenBase::point_type;
  using typename IntBase::cell_ctx_type;

  // shorthands to save typing
  using S = typename GenBase::size_type;
  using T = typename GenBase::value_type;

 protected:
  // load in Base members
  using GenBase::envelope_;
  using IntBase::grid0_;
  using IntBase::nblocks_;
  using IntBase::ndim_;
  using IntBase::ndiv0_;
  using IntBase::ndiv1_;
  using IntBase::ndiv2_;
  using IntBase::order_;
  using IntBase::ordered_grid0_;
  using IntBase::ordered_grid_;

 public:
  /*!
   * @brief construct a new BasinGenerator
   *
   * @param ndim number of dimensions
   * @param ndiv1 number of coarse grid divisions along each dimension
   * @param ndiv2 number of fine grid divisions along each dimension
   */
  explicit BasinGenerator(S ndim, S ndiv1 = 8, S ndiv2 = 16) : GenBase(ndim, ndiv1, ndiv2) {}

 protected:
  /// @name Envelope hooks used by `GeneratorBase`
  /// @{

  /// @brief cell-context buffer for `map_point`
  [[nodiscard]] inline cell_ctx_type make_cell_ctx() const {
    return cell_ctx_type({ndim_, 3});
  }

  /// @brief one `{ndiv1, ndiv2}` block per order entry (see the class description)
  [[nodiscard]] std::vector<S> env_shape() const {
    return {ndim_, ndiv1_, ndiv2_};
  }

  /*!
   * @brief flat indices of the raw factor of each order entry
   *
   * A diagonal entry uses its cell `ig0`, a conditional entry `(ig1, ig2)`
   * with the coarse cell of the parent `ig1 = ig0(idim1) / ndiv2` and its own
   * conditional cell `ig2`.
   */
  void env_indices(const cell_ctx_type& cell, std::span<S> indices) const {
    for (S iord = 0; iord < ndim_; ++iord) {
      const S ig0 = cell(order_(iord, 0), 0);
      const S offset = iord < nblocks_ ? ig0 : (ig0 / ndiv2_) * ndiv2_ + cell(order_(iord, 1), 2);
      indices[iord] = iord * ndiv0_ + offset;
    }
  }

  /// @brief the envelope layout depends on the sampling order as well as the grid edges
  [[nodiscard]] util::HashValue_t env_grid_hash() const {
    return this->hash().add(order_.data(), order_.size()).value();
  }

  /*!
   * @brief draw a point with density R/V and map it through the grids
   *
   * Walks through the sampling order with one CDF search per dimension. The
   * position inside the drawn CDF segment is uniform and the BASIN map is
   * linear inside each cell, so the position goes straight to x-space without
   * a detour through u. `point.x` and `point.weight` are set as `map_point`
   * would for the drawn cells. The coarse cells the children are conditioned
   * on come from the walk itself (no search in x-space).
   *
   * @param point output point; coordinates and weight are overwritten
   * @return the raw product R(u), conditional dimensions included
   */
  inline T env_propose(point_type& point) {
    return propose_impl<false>(point, nullptr);
  }

  /*!
   * @brief like `env_propose(point)`, but also record the drawn cells in `cell`
   *        (as `map_point` would) so the factors at `env_indices(cell)` multiply to R
   *
   * A point on a coarse edge lies in two coarse cells; for a dimension with
   * children, the recorded `ig0` is then the one inside the coarse cell of the walk.
   */
  inline T env_propose(point_type& point, cell_ctx_type& cell) {
    return propose_impl<true>(point, &cell);
  }

  /// @brief compute the subtree integrals bottom-up and build the piecewise-constant CDFs
  ///
  /// A conditional row is only split at its own cell edges and, if the dimension
  /// has children, at its coarse cell edges. That gives at most ndiv1+ndiv2-1
  /// intervals per row, no matter how deep the tree is.
  ///
  /// @return the integral V of the envelope R and the proposal caches
  auto env_prepare(const ndarray::NDArray<T, S>& table) const {
    EnvelopeCache cache{
        .root_cdf = ndarray::NDArray<T, S>({ndim_, ndiv0_}),
        .conditional_rows = std::vector<ConditionalRow>(static_cast<std::size_t>(ndim_) * ndiv1_),
        .child_integrals = ndarray::NDArray<T, S>({ndim_, ndiv1_}),
        .has_children = std::vector<uint8_t>(ndim_, 0),
        .walk_ig1 = std::vector<S>(ndim_, 0)};
    cache.child_integrals.fill(T(1));
    const auto max_segments = static_cast<std::size_t>(ndiv1_) + ndiv2_ - 1;
    // naming: `ig1` is the coarse cell of the parent `idim1` (the row), `ig2` the
    // conditional cell of `idim2` and `ig1_idim2` the coarse cell of `idim2` itself
    // (the row of its children). `x_*` are edges in x-space, `u_*` in u-space; the
    // `*_seg_*` edges delimit a segment, the plain ones the conditional cell `ig2`.
    for (S iord = ndim_; iord-- > nblocks_;) {
      const S idim1 = order_(iord, 0);
      const S idim2 = order_(iord, 1);
      for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
        auto& row = cache.conditional_rows[row_index(iord, ig1)];
        row.cdf.reserve(max_segments);
        row.segments.reserve(max_segments);
        S ig1_idim2 = 0;
        T u_low = T(0);  // lower edge of the next segment in u-space
        for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
          const T x_low = ig2 == 0 ? T(0) : ordered_grid_(iord, ig1, ig2 - 1);
          const T x_upp = ordered_grid_(iord, ig1, ig2);
          if (!(x_upp > x_low)) {
            throw std::runtime_error("conditional grid has a zero-width cell");
          }
          T x_seg_low = x_low;
          do {
            T x_seg_upp = x_upp;
            if (cache.has_children[idim2]) {
              while (ig1_idim2 + 1 < ndiv1_ && coarse_edge(idim2, ig1_idim2) <= x_seg_low)
                ++ig1_idim2;
              x_seg_upp = std::min(x_seg_upp, coarse_edge(idim2, ig1_idim2));
            }
            const T u_upp = x_seg_upp == x_upp
                                ? T(ig2 + 1) / T(ndiv2_)
                                : (T(ig2) + (x_seg_upp - x_low) / (x_upp - x_low)) / T(ndiv2_);
            const T factor = table(iord, ig1, ig2);
            const T height = factor * cache.child_integrals(idim2, ig1_idim2);
            const T sum = row.cdf.empty() ? T(0) : row.cdf.back();
            row.cdf.push_back(detail::cdf_step(sum, height * (u_upp - u_low)));
            row.segments.push_back({x_seg_low, x_seg_upp - x_seg_low, T(ndiv2_) * (x_upp - x_low),
                                    factor, ig2, ig1_idim2});
            u_low = u_upp;
            x_seg_low = x_seg_upp;
          } while (x_seg_low < x_upp);
        }
        cache.child_integrals(idim1, ig1) *= row.cdf.back();
      }
      cache.has_children[idim1] = 1;
    }
    T volume = T(1);
    for (S iord = 0; iord < nblocks_; ++iord) {
      const S idim0 = order_(iord, 0);
      T sum = T(0);
      for (S ig0 = 0; ig0 < ndiv0_; ++ig0) {
        const T factor = table[iord * ndiv0_ + ig0];
        sum = detail::cdf_step(sum, factor * cache.child_integrals(idim0, ig0 / ndiv2_));
        cache.root_cdf(iord, ig0) = sum;
      }
      volume *= sum / T(ndiv0_);
    }
    return std::pair{volume, std::move(cache)};
  }

  /// @}

 private:
  // an interval of a conditional row on which the product of the child
  // integrals is constant; the map is linear on it
  struct Segment {
    T x_low;      // lower edge in x-space
    T x_width;    // width in x-space
    T jacobian;   // dx/du of the grid cell that contains the segment
    T factor;     // raw envelope factor `envelope_(iord, ig1, ig2)`
    S ig2;        // conditional cell of `idim2` (column in the raw conditional table)
    S ig1_idim2;  // coarse cell of `idim2` in x-space, the row of its children (unused for leaves)
  };
  struct ConditionalRow {
    std::vector<T> cdf;  // cumulative mass in u-space; the last entry is the subtree integral
    std::vector<Segment> segments;
  };
  // everything the proposal needs, rebuilt from the raw table whenever it is sealed
  struct EnvelopeCache {
    ndarray::NDArray<T, S> root_cdf;
    std::vector<ConditionalRow> conditional_rows;
    // F_j(ig1): product of the subtree integrals of the children for coarse cell ig1 of j (x-space)
    ndarray::NDArray<T, S> child_integrals;
    std::vector<uint8_t> has_children;
    // scratch: coarse cell of each dimension drawn in the walk; the children read it as their `ig1`
    std::vector<S> walk_ig1;
  };

  void env_commit_cache(EnvelopeCache&& cache) noexcept {
    envelope_cache_ = std::move(cache);
  }

  // one conditional row per (order entry, coarse cell of the parent)
  [[nodiscard]] std::size_t row_index(S iord, S ig1) const noexcept {
    return static_cast<std::size_t>(iord) * ndiv1_ + ig1;
  }

  ConditionalRow& conditional_row(S iord, S ig1) {
    return envelope_cache_.conditional_rows[row_index(iord, ig1)];
  }

  // upper edge in x-space of the coarse cell `ig1` of dimension `idim`
  T coarse_edge(S idim, S ig1) const {
    return grid0_(idim, (ig1 + 1) * ndiv2_ - 1);
  }

  template <bool RecordCells>
  T propose_impl(point_type& point, [[maybe_unused]] cell_ctx_type* cell) {
    assert(point.x.size() == static_cast<std::size_t>(ndim_));
    T envelope = T(1);
    T weight = T(1);
    for (S iord = 0; iord < nblocks_; ++iord) {
      const S idim0 = order_(iord, 0);
      const auto draw =
          detail::sample_cdf<T, S>({&envelope_cache_.root_cdf(iord, 0), ndiv0_}, IntBase::ran());
      const S ig0 = draw.cell;
      const T x_low = ig0 > 0 ? ordered_grid0_(iord, ig0 - 1) : T(0);
      const T x_upp = ordered_grid0_(iord, ig0);
      point.x[idim0] = x_low + draw.fraction * (x_upp - x_low);
      weight *= T(ndiv0_) * (x_upp - x_low);
      envelope *= envelope_[iord * ndiv0_ + ig0];
      envelope_cache_.walk_ig1[idim0] = ig0 / ndiv2_;
      if constexpr (RecordCells) {
        (*cell)(idim0, 0) = ig0;
        (*cell)(idim0, 1) = idim0;
        (*cell)(idim0, 2) = ndiv2_;
      }
    }  // for iord
    for (S iord = nblocks_; iord < ndim_; ++iord) {
      const S idim1 = order_(iord, 0);
      const S idim2 = order_(iord, 1);
      const S ig1 = envelope_cache_.walk_ig1[idim1];
      const auto& row = conditional_row(iord, ig1);
      const auto draw = detail::sample_cdf<T, S>(row.cdf, IntBase::ran());
      const Segment& segment = row.segments[draw.cell];
      point.x[idim2] = segment.x_low + draw.fraction * segment.x_width;
      weight *= segment.jacobian;
      envelope *= segment.factor;
      envelope_cache_.walk_ig1[idim2] = segment.ig1_idim2;
      if constexpr (RecordCells) {
        const T* row0 = &grid0_(idim2, 0);
        S ig0 = static_cast<S>(std::lower_bound(row0, row0 + ndiv0_, point.x[idim2]) - row0);
        if (envelope_cache_.has_children[idim2]) {
          ig0 = std::clamp(ig0, segment.ig1_idim2 * ndiv2_, (segment.ig1_idim2 + 1) * ndiv2_ - 1);
        }
        (*cell)(idim2, 0) = ig0;
        (*cell)(idim2, 1) = idim1;
        (*cell)(idim2, 2) = segment.ig2;
      }
    }  // for iord
    point.weight = weight;
    return envelope;
  }

  // only the raw table gets serialized; proposals are rebuilt on load
  EnvelopeCache envelope_cache_;
};

}  // namespace kakuhen::integrator
