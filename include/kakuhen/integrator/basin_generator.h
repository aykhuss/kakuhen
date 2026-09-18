#pragma once

#include "kakuhen/integrator/basin.h"
#include "kakuhen/integrator/generator_base.h"
#include "kakuhen/ndarray/ndarray.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace kakuhen::integrator {

/*!
 * @brief BASIN integrator with event generation.
 *
 * `GeneratorBase` takes care of the generation itself and this class only
 * defines the envelope. It follows the frozen BASIN sampling structure:
 *
 * - each diagonal dimension (entries `iord < nblocks` of the sampling order)
 *   has a 1D table over its `ndiv0` u-cells, the same cells that `map_point`
 *   picks;
 * - each conditional dimension has a table over `(ig1, ig2)`, where `ig1` is
 *   the coarse cell of the parent dimension and `ig2` the conditional u-cell.
 *   This is the same nesting as in `ordered_grid_` that the map samples from.
 *
 * Since `ndiv0 = ndiv1 * ndiv2`, both kinds of tables fit into a single
 * `{ndim, ndiv1, ndiv2}` array indexed by the order entry. The diagonal rows
 * are accessed through a `{ndim, ndiv0}` reshaped view, in the same way as
 * `ordered_grid_`/`ordered_grid0_` in `Basin`.
 *
 * The envelope R is always the raw product of table entries. When the
 * envelope is sealed, the subtree integrals are computed bottom-up from the
 * children to their parents. Conditional cells are split at the coarse-grid
 * edges in x-space, so that the product of the child integrals is constant on
 * each interval. The masses of these intervals in u-space are exact and give
 * the cached CDFs, so there is no maximization or integration needed during
 * generation. A proposal walks through the sampling order to sample
 * q = R / int R du with one CDF search per dimension and returns the raw
 * product R, the same one used in the raising passes.
 *
 * @tparam NT The numeric traits for the integrator.
 * @tparam RNG The random number generator to use.
 * @tparam DIST The random number distribution to use.
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
   * @brief Construct a new BasinGenerator object.
   *
   * @param ndim The number of dimensions of the integration.
   * @param ndiv1 The number of divisions for the coarse grid along each dimension.
   * @param ndiv2 The number of divisions for the fine grid along each dimension.
   */
  explicit BasinGenerator(S ndim, S ndiv1 = 8, S ndiv2 = 16) : GenBase(ndim, ndiv1, ndiv2) {}

 protected:
  /// @name Envelope hooks used by `GeneratorBase`
  /// @{

  /// @brief Cell-context buffer for the `map_point` of this integrator.
  [[nodiscard]] inline cell_ctx_type make_cell_ctx() const {
    return cell_ctx_type({ndim_, 3});
  }

  /// @brief One `{ndiv1, ndiv2}` block per order entry (see the class description).
  [[nodiscard]] std::vector<S> env_shape() const {
    return {ndim_, ndiv1_, ndiv2_};
  }

  /*!
   * @brief Evaluate the envelope R using the cells recorded by `map_point`.
   *        The same product is used for raising and for generation.
   *
   * For each order entry, a diagonal dimension looks up its cell `ig0`, while
   * a conditional dimension looks up `(ig1, ig2)` with the coarse cell of the
   * parent `ig1 = ig0(idim1) / ndiv2` and its own conditional cell `ig2`.
   */
  [[nodiscard]] inline T env_value(const cell_ctx_type& cell) const {
    assert(cell.size() == static_cast<std::size_t>(ndim_) * 3);
    T envelope = T(1);
    for (S iord = 0; iord < ndim_; ++iord) {
      if (iord < nblocks_) {
        const S idim0 = order_(iord, 0);
        envelope *= env0_(iord, cell(idim0, 0));
      } else {
        const S idim1 = order_(iord, 0);
        const S idim2 = order_(iord, 1);
        assert(cell(idim2, 1) == idim1);
        envelope *= envelope_(iord, cell(idim1, 0) / ndiv2_, cell(idim2, 2));
      }
    }  // for iord
    return envelope;
  }

  /// @brief Multiply the cells recorded in `cell` (one per order entry) by `gamma`.
  ///        These are the same cells that `env_value` reads.
  inline void env_raise(const cell_ctx_type& cell, T gamma) {
    assert(cell.size() == static_cast<std::size_t>(ndim_) * 3);
    for (S iord = 0; iord < ndim_; ++iord) {
      if (iord < nblocks_) {
        const S idim0 = order_(iord, 0);
        env0_(iord, cell(idim0, 0)) *= gamma;
      } else {
        const S idim1 = order_(iord, 0);
        const S idim2 = order_(iord, 1);
        envelope_(iord, cell(idim1, 0) / ndiv2_, cell(idim2, 2)) *= gamma;
      }
    }  // for iord
  }

  /*!
   * @brief Draw a point with density R/V and map it through the grids.
   *
   * Walks through the sampling order with one CDF search per dimension. The
   * position inside the drawn CDF segment is uniform, and the BASIN map is
   * linear inside each cell, so the position goes straight to x-space without
   * a detour through u. `point.x` and `point.weight` are set as `map_point`
   * would set them for the drawn cells. The coarse cells that the children
   * are conditioned on come from the walk itself (no search in x-space).
   *
   * @param point Output point whose coordinates and weight are overwritten.
   * @return The raw product R(u), also for the conditional dimensions.
   */
  inline T env_propose(point_type& point) {
    return propose_impl<false>(point, nullptr);
  }

  /*!
   * @brief As `env_propose(point)`, and also record the drawn cells in `cell`
   *        (as `map_point` would), such that `env_value(cell)` gives back R.
   *
   * A point on a coarse edge lies in two coarse cells; the recorded `ig0` of a
   * dimension with children is then the one inside the coarse cell of the walk.
   */
  inline T env_propose(point_type& point, cell_ctx_type& cell) {
    return propose_impl<true>(point, &cell);
  }

  /// @brief Compute the subtree integrals bottom-up and build the piecewise-constant CDFs.
  /// A conditional row is only split at its own cell edges and, if the dimension
  /// has children, at its coarse cell edges. That gives at most ndiv1+ndiv2-1
  /// intervals per row, no matter how deep the tree is.
  /// @return The integral V of the envelope R.
  T env_prepare() {
    env_rebuild_storage();
    child_integrals_.fill(T(1));
    std::fill(has_children_.begin(), has_children_.end(), uint8_t(0));
    // naming: `ig1` is the coarse cell of the parent `idim1` (the row), `ig2` the
    // conditional cell of `idim2` and `ig1_idim2` the coarse cell of `idim2` itself
    // (the row of its children). `x_*` are edges in x-space, `u_*` in u-space; the
    // `*_seg_*` edges delimit a segment, the plain ones the conditional cell `ig2`.
    for (S iord = ndim_; iord-- > nblocks_;) {
      const S idim1 = order_(iord, 0);
      const S idim2 = order_(iord, 1);
      for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
        auto& row = conditional_row(iord, ig1);
        row.cdf.clear();
        row.segments.clear();
        S ig1_idim2 = 0;
        T u_low = T(0);  // lower edge of the next segment in u-space
        for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
          const T x_low = ig2 == 0 ? T(0) : ordered_grid_(iord, ig1, ig2 - 1);
          const T x_upp = ordered_grid_(iord, ig1, ig2);
          if (!(x_upp > x_low)) {
            throw std::runtime_error("BasinGenerator: conditional grid has a zero-width cell");
          }
          T x_seg_low = x_low;
          do {
            T x_seg_upp = x_upp;
            if (has_children_[idim2]) {
              while (ig1_idim2 + 1 < ndiv1_ && coarse_edge(idim2, ig1_idim2) <= x_seg_low)
                ++ig1_idim2;
              x_seg_upp = std::min(x_seg_upp, coarse_edge(idim2, ig1_idim2));
            }
            const T u_upp = x_seg_upp == x_upp
                                ? T(ig2 + 1) / T(ndiv2_)
                                : (T(ig2) + (x_seg_upp - x_low) / (x_upp - x_low)) / T(ndiv2_);
            const T factor = envelope_(iord, ig1, ig2);
            const T height = factor * child_integrals_(idim2, ig1_idim2);
            const T sum = row.cdf.empty() ? T(0) : row.cdf.back();
            row.cdf.push_back(detail::cdf_step(sum, height * (u_upp - u_low)));
            row.segments.push_back({x_seg_low, x_seg_upp - x_seg_low, T(ndiv2_) * (x_upp - x_low),
                                    factor, ig2, ig1_idim2});
            u_low = u_upp;
            x_seg_low = x_seg_upp;
          } while (x_seg_low < x_upp);
        }
        child_integrals_(idim1, ig1) *= row.cdf.back();
      }
      has_children_[idim1] = 1;
    }
    T volume = T(1);
    for (S iord = 0; iord < nblocks_; ++iord) {
      const S idim0 = order_(iord, 0);
      T sum = T(0);
      for (S ig0 = 0; ig0 < ndiv0_; ++ig0) {
        sum = detail::cdf_step(sum, env0_(iord, ig0) * child_integrals_(idim0, ig0 / ndiv2_));
        root_cdf_(iord, ig0) = sum;
      }
      volume *= sum / T(ndiv0_);
    }
    return volume;
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

  ConditionalRow& conditional_row(S iord, S ig1) {
    return conditional_rows_[static_cast<std::size_t>(iord) * ndiv1_ + ig1];
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
      const auto draw = detail::sample_cdf<T, S>({&root_cdf_(iord, 0), ndiv0_}, IntBase::ran());
      const S ig0 = draw.cell;
      const T x_low = ig0 > 0 ? ordered_grid0_(iord, ig0 - 1) : T(0);
      const T x_upp = ordered_grid0_(iord, ig0);
      point.x[idim0] = x_low + draw.fraction * (x_upp - x_low);
      weight *= T(ndiv0_) * (x_upp - x_low);
      envelope *= env0_(iord, ig0);
      walk_ig1_[idim0] = ig0 / ndiv2_;
      if constexpr (RecordCells) {
        (*cell)(idim0, 0) = ig0;
        (*cell)(idim0, 1) = idim0;
        (*cell)(idim0, 2) = ndiv2_;
      }
    }  // for iord
    for (S iord = nblocks_; iord < ndim_; ++iord) {
      const S idim1 = order_(iord, 0);
      const S idim2 = order_(iord, 1);
      const S ig1 = walk_ig1_[idim1];
      const auto& row = conditional_row(iord, ig1);
      const auto draw = detail::sample_cdf<T, S>(row.cdf, IntBase::ran());
      const Segment& segment = row.segments[draw.cell];
      point.x[idim2] = segment.x_low + draw.fraction * segment.x_width;
      weight *= segment.jacobian;
      envelope *= segment.factor;
      walk_ig1_[idim2] = segment.ig1_idim2;
      if constexpr (RecordCells) {
        const T* row0 = &grid0_(idim2, 0);
        S ig0 = static_cast<S>(std::lower_bound(row0, row0 + ndiv0_, point.x[idim2]) - row0);
        if (has_children_[idim2]) {
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

  // the base may have replaced the table since the last seal: update the
  // diagonal view and resize the caches if the grid shape changed.
  void env_rebuild_storage() {
    env0_ = envelope_.reshape({ndim_, ndiv0_});
    if (conditional_rows_.size() == static_cast<std::size_t>(ndim_) * ndiv1_ &&
        std::ranges::equal(root_cdf_.shape(), std::array{ndim_, ndiv0_}) &&
        std::ranges::equal(child_integrals_.shape(), std::array{ndim_, ndiv1_})) {
      return;
    }
    root_cdf_ = ndarray::NDArray<T, S>({ndim_, ndiv0_});
    child_integrals_ = ndarray::NDArray<T, S>({ndim_, ndiv1_});
    conditional_rows_.clear();
    conditional_rows_.resize(static_cast<std::size_t>(ndim_) * ndiv1_);
    for (auto& row : conditional_rows_) {
      row.cdf.reserve(static_cast<std::size_t>(ndiv1_) + ndiv2_ - 1);
      row.segments.reserve(static_cast<std::size_t>(ndiv1_) + ndiv2_ - 1);
    }
    has_children_.assign(ndim_, 0);
    walk_ig1_.assign(ndim_, S(0));
  }

  // only the raw table (stored in GeneratorBase) gets serialized;
  // everything needed for the proposals is rebuilt when sealing.
  ndarray::NDView<T, S> env0_;
  ndarray::NDArray<T, S> root_cdf_;
  std::vector<ConditionalRow> conditional_rows_;
  // F_j(ig1): product of the subtree integrals of the children for coarse cell ig1 of j (x-space)
  ndarray::NDArray<T, S> child_integrals_;
  std::vector<uint8_t> has_children_;
  // coarse cell of each dimension drawn in the walk; the children read it as their `ig1`
  std::vector<S> walk_ig1_;
};

}  // namespace kakuhen::integrator
