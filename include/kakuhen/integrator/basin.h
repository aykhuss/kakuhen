#pragma once

#include "kakuhen/integrator/detail/grid_io.h"
#include "kakuhen/integrator/grid_accumulator.h"
#include "kakuhen/integrator/integrator_base.h"
#include "kakuhen/ndarray/ndarray.h"
#include "kakuhen/ndarray/ndview.h"
#include "kakuhen/util/algorithm.h"
#include "kakuhen/util/hash.h"
#include "kakuhen/util/math.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace kakuhen::integrator {

/*!
 * @brief An integrator based on Blockwise Adaptive Sampling with Interdimensional Nesting (BASIN).
 *
 * This class implements a sophisticated adaptive Monte Carlo integration algorithm
 * that models correlations between integration dimensions. It uses a nested
 * grid structure where each dimension's grid is conditioned on the value of
 * another, allowing it to adapt to complex integrand shapes.
 *
 * Each adaptation also updates the sampling order, a forest of dimension
 * dependencies, from local variance gains predicted by the grid data.
 *
 * @tparam NT The numeric traits for the integrator.
 * @tparam RNG The random number generator to use.
 * @tparam DIST The random number distribution to use.
 */
template <typename NT = util::num_traits_t<>,
          typename RNG = typename IntegratorDefaults<NT>::rng_type,
          typename DIST = typename IntegratorDefaults<NT>::dist_type>
class Basin : public IntegratorBase<Basin<NT, RNG, DIST>, NT, RNG, DIST> {
 public:
  static constexpr IntegratorId static_id() noexcept {
    return IntegratorId::BASIN;
  }

  // dependent class: need to explicitly load things from the Base
  using Base = IntegratorBase<Basin<NT, RNG, DIST>, NT, RNG, DIST>;
  using typename Base::count_type;
  using typename Base::int_acc_type;
  using typename Base::num_traits;
  using typename Base::point_type;
  using typename Base::seed_type;
  using typename Base::size_type;
  using typename Base::value_type;

  // shorthands to save typing
  using S = typename Base::size_type;
  using T = typename Base::value_type;
  using U = typename Base::count_type;
  using grid_acc_type = GridAccumulator<T, U>;

  //  member variables
  using Base::ndim_;
  using Base::opts_;

  /*!
   * @brief The cell context type recorded by `map_point`; shape `{ndim, 3}`.
   *
   * For each dimension the three entries are `(ig0, cond, ig2)`:
   * - `ig0`: the diagonal cell index in `[0, ndiv0)`;
   * - `cond`: the conditioning dimension (the dimension itself if sampled
   *   diagonally);
   * - `ig2`: the conditional sub-grid cell in `[0, ndiv2)` (the sentinel
   *   `ndiv2` if sampled diagonally).
   */
  using cell_ctx_type = ndarray::NDArray<S, S>;

  /*!
   * @brief Construct a new Basin object.
   *
   * @param ndim The number of dimensions of the integration.
   * @param ndiv1 The number of divisions for the coarse grid along each dimension.
   * @param ndiv2 The number of divisions for the fine grid along each dimension.
   */
  explicit Basin(S ndim, S ndiv1 = 8, S ndiv2 = 16)
      : Base(ndim),
        ndiv1_{ndiv1},
        ndiv2_{ndiv2},
        ndiv0_{S(std::uintmax_t(ndiv1) * ndiv2)},  // a narrow S promotes to int, which can overflow
        grid_({ndim, ndim, ndiv1, ndiv2}),
        order_({ndim, 2}),
        ordered_grid_({ndim, ndiv1, ndiv2}),
        accumulator_count_{0},
        accumulator_({ndim, ndim, ndiv1, ndiv2}) {
    assert(ndim > 0 && ndiv1 > 1 && ndiv2 > 1);
    grid0_ = grid_.reshape({ndim_, ndim_, ndiv0_}).diagonal(0, 1);
    accumulator0_ = accumulator_.reshape({ndim_, ndim_, ndiv0_}).diagonal(0, 1);
    ordered_grid0_ = ordered_grid_.reshape({ndim_, ndiv0_});
    reset();
  };

  /*!
   * @brief Construct a new Basin object by loading its state from a file.
   *
   * @param filepath The path to the file containing the integrator's state.
   */
  explicit Basin(const std::filesystem::path& filepath) : Base(0) {
    Base::load(filepath);
  }

  /*!
   * @brief Get the number of grid divisions in the first dimension.
   *
   * @return The value of the number of grid divisions for dim 1.
   */
  [[nodiscard]] inline S ndiv1() const noexcept {
    return ndiv1_;
  }
  /*!
   * @brief Get the number of grid divisions in the second dimension.
   *
   * @return The value of the number of grid divisions for dim 2.
   */
  [[nodiscard]] inline S ndiv2() const noexcept {
    return ndiv2_;
  }
  /*!
   * @brief Get the number of grid divisions in the diagonal dimension.
   *
   * @return The value of the number of grid divisions for projections.
   */
  [[nodiscard]] inline S ndiv0() const noexcept {
    return ndiv0_;
  }
  /*!
   * @brief Get the number of uncorrelated blocks in the learned sampling order.
   *
   * Equals `ndim()` when no inter-dimensional correlations were detected; each
   * remaining dimension is sampled conditionally on another dimension.
   *
   * @return The number of diagonally sampled dimensions.
   */
  [[nodiscard]] inline S nblocks() const noexcept {
    return nblocks_;
  }

  /*!
   * @brief Set the alpha parameter for grid adaptation.
   *
   * @param alpha The new value for the alpha parameter; must be finite and >= 0.
   * @throws std::invalid_argument if `alpha` is negative or not finite.
   */
  inline void set_alpha(const T& alpha) {
    if (!std::isfinite(alpha) || alpha < T(0)) {
      throw std::invalid_argument("alpha must be finite and >= 0");
    }
    alpha_ = alpha;
  }
  /*!
   * @brief Get the alpha parameter for grid adaptation.
   *
   * @return The current value of the alpha parameter.
   */
  [[nodiscard]] inline T alpha() const noexcept {
    return alpha_;
  }

  /*!
   * @brief Set the weight for smoothing the grid adaptation.
   *
   * Below 2 the smoothing kernel `[1, w, 1]` flips an odd-even pattern of the
   * bin widths, so the adaptation amplifies it instead of damping it.
   *
   * @param weight_smooth The new value for the weight smoothing parameter;
   *        must be finite and >= 2.
   * @throws std::invalid_argument if `weight_smooth` is below 2 or not finite.
   */
  inline void set_weight_smooth(const T& weight_smooth) {
    if (!std::isfinite(weight_smooth) || weight_smooth < T(2)) {
      throw std::invalid_argument("weight_smooth must be finite and >= 2");
    }
    weight_smooth_ = weight_smooth;
  }
  /*!
   * @brief Get the weight for smoothing the grid adaptation.
   *
   * @return The current value of the weight smoothing parameter.
   */
  [[nodiscard]] inline T weight_smooth() const noexcept {
    return weight_smooth_;
  }

  /*!
   * @brief set the minimum local variance gain to change the sampling order
   *
   * @param min_gain fraction of current variance, required for candidate moves and the
   *                 completed forest; >= 0, infinity keeps the order fixed
   * @throws std::invalid_argument if `min_gain` is negative or NaN
   */
  inline void set_min_gain(const T& min_gain) {
    if (!(min_gain >= T(0))) throw std::invalid_argument("min_gain must be >= 0");
    min_gain_ = min_gain;
  }
  /// @brief minimum local variance gain to change the sampling order
  [[nodiscard]] inline T min_gain() const noexcept {
    return min_gain_;
  }

  /*!
   * @brief Computes a hash of the current grid state.
   *
   * @return A `kakuhen::util::Hash` object representing the state of the grid.
   */
  [[nodiscard]] inline kakuhen::util::Hash hash() const {
    return kakuhen::util::Hash().add(ndim_).add(ndiv1_).add(ndiv2_).add(grid_.data(), grid_.size());
  }

  /// @brief fingerprint of the sampling map (grid and order)
  [[nodiscard]] inline kakuhen::util::HashValue_t map_hash() const {
    return hash().add(order_.data(), order_.size()).value();
  }

  /*!
   * @brief Generates a prefix string for filenames.
   *
   * @param with_hash If true, the hash of the grid is included in the prefix.
   * @return A prefix string for filenames.
   */
  [[nodiscard]] inline std::string prefix(bool with_hash = false) const noexcept {
    if (with_hash) return std::format("basin_{}d_{}", ndim_, hash().encode_hex());
    return std::format("basin_{}d", ndim_);
  }

 private:
  friend Base;  // only integrate() enters an iteration
  using typename Base::ProgressTracker;

  /// @name Integration Implementation
  /// @{

  /*!
   * @brief Implementation of the integration loop for a single iteration.
   *
   * @tparam I The type of the integrand function.
   * @tparam ProgressCb The type of the progress callback (or std::nullptr_t).
   * @param integrand The function to integrate. Only invoked, never consumed, hence
   *        taken by lvalue reference (the caller invokes it once per iteration).
   * @param neval The number of evaluations to perform.
   * @param tracker Per-call progress bookkeeping shared with the base class.
   * @param progress_cb The progress callback for milestone notifications.
   * @return An `int_acc_type` containing the accumulated results for this iteration.
   */
  template <typename I, typename ProgressCb = std::nullptr_t>
  int_acc_type integrate_impl(I& integrand, U neval, [[maybe_unused]] ProgressTracker& tracker,
                              [[maybe_unused]] ProgressCb&& progress_cb = nullptr) {
    int_acc_type result;

    Point<num_traits> point{ndim_, opts_.user_data.value_or(nullptr)};
    std::vector<T> u_buf(ndim_);
    cell_ctx_type cell({ndim_, 3});
    // per dimension b: grid data G = (f w)^2 / w_b
    std::vector<T> g(ndim_);

    const bool skip_accum = opts_.frozen.value_or(false);
    const bool strict_finite = opts_.strict_finite_integrand.value_or(false);

    for (U i = 0; i < neval; ++i) {
      for (S idim = 0; idim < ndim_; ++idim)
        u_buf[idim] = Base::ran();
      point.sample_index = i;
      map_point_impl(u_buf, point, cell, !skip_accum);
      const T fval = point.weight * integrand(point);
      if (strict_finite && !std::isfinite(fval)) {
        throw std::runtime_error("integrate: non-finite integrand contribution");
      }
      const T fval2 = fval * fval;
      result.accumulate(fval, fval2);
      // pending stats keep every sample in step with the grid, so snapshots stay consistent
      if (!skip_accum) result_.accumulate(fval, fval2);
      // a non-finite sample shows up in the result but must not corrupt the grid
      if (!skip_accum && std::isfinite(fval2)) {
        // divide out b's sampled factor to compare candidate factors locally
        for (S b = 0; b < ndim_; ++b) {
          const T w = sampled_weight(b, cell);
          g[b] = fval2 > T(0) ? fval2 / w : T(0);  // w = 0 implies f w = 0
        }
        accumulator_count_++;
        collect_grid(point, cell, g);
      }

      if constexpr (is_progress_callback_v<ProgressCb>) {
        if (Base::check_eval_milestone(tracker, progress_cb, i)) break;
      }
    }  // for i

    return result;
  }

 public:
  /*!
   * @brief Maps caller-supplied uniform coordinates through the BASIN grids.
   *
   * Each `u[idim]` drives the physical dimension `idim`. Diagonal dimensions are
   * mapped through the ordered one-dimensional grids, then conditional
   * dimensions are mapped through the ordered nested grids. The mapped
   * coordinates and importance-sampling Jacobian are written to `point`, while
   * row `idim` of `cell` receives the per-dimension context `(ig0, cond, ig2)`:
   * the diagonal `ig0` cell index, the conditioning dimension `cond` (the
   * dimension itself for diagonally sampled dimensions), and the conditional
   * sub-grid cell `ig2` (the sentinel `ndiv2` for diagonally sampled
   * dimensions). This method does not draw from the RNG, does not mutate
   * integrator state, and leaves `point.sample_index` unchanged.
   *
   * @param u Uniform randoms in [0, 1), one per physical dimension.
   * @param point Output point whose coordinates and weight are overwritten.
   * @param cell Output cell context; must have shape `{ndim(), 3}`.
   * @param all_cells if false, `ig0` of conditional dimensions without children is left at the
   *        sentinel `ndiv0` (saves their lookup in the diagonal grid)
   */
  inline void map_point_impl(std::span<const T> u, Point<num_traits>& point, cell_ctx_type& cell,
                             bool all_cells = true) const {
    assert(cell.size() == static_cast<std::size_t>(ndim_) * 3);
    point.weight = T(1);

    /// (a) diagonal map
    for (S iord = 0; iord < nblocks_; ++iord) {
      assert(order_(iord, 0) == order_(iord, 1));
      const S idim0 = order_(iord, 0);
      //> intervals u in [ i/ndiv0_ , (i+1)/ndiv0_ ] mapped to i
      const T scaled = u[idim0] * T(ndiv0_);
      const S ig0 = S(scaled);
      assert(ig0 >= 0 && ig0 < ndiv0_);
      //> reuse the rounded product for both the bin and the local coordinate in [0, 1)
      const T rand = scaled - T(ig0);
      assert(rand >= T(0) && rand < T(1));
      const T x_low = ig0 > 0 ? ordered_grid0_(iord, ig0 - 1) : T(0);
      const T x_upp = ordered_grid0_(iord, ig0);
      point.x[idim0] = x_low + rand * (x_upp - x_low);
      // point.x[idim0] = x_low * (T(1) - rand) + x_upp * rand;
      point.weight *= T(ndiv0_) * (x_upp - x_low);
      cell(idim0, 0) = ig0;
      cell(idim0, 1) = idim0;
      cell(idim0, 2) = ndiv2_;
    }  // for iord

    /// (b) conditional map
    for (S iord = nblocks_; iord < ndim_; ++iord) {
      assert(order_(iord, 0) != order_(iord, 1));
      const S idim1 = order_(iord, 0);
      const S idim2 = order_(iord, 1);
      /// check that the 1st dimension is set properly
      assert(point.x[idim1] >= T(0) && point.x[idim1] <= T(1));
      // the first child looks up the diagonal cell of its parent
      S& ig0_1 = cell(idim1, 0);
      if (ig0_1 == ndiv0_) ig0_1 = diag_cell(idim1, point.x[idim1]);
      const S ig1 = ig0_1 / ndiv2_;
      assert(ig1 >= 0 && ig1 < ndiv1_);
      //> intervals u in [ i/ndiv2_ , (i+1)/ndiv2_ ] mapped to i
      const T scaled = u[idim2] * T(ndiv2_);
      const S ig2 = S(scaled);
      assert(ig2 >= 0 && ig2 < ndiv2_);
      //> reuse the rounded product for both the bin and the local coordinate in [0, 1)
      const T rand = scaled - T(ig2);
      assert(rand >= T(0) && rand < T(1));
      const T x_low = ig2 > 0 ? ordered_grid_(iord, ig1, ig2 - 1) : T(0);
      const T x_upp = ordered_grid_(iord, ig1, ig2);
      const T x = x_low + rand * (x_upp - x_low);
      point.x[idim2] = x;
      // point.x[idim2] = x_low * (T(1) - rand) + x_upp * rand;
      point.weight *= T(ndiv2_) * (x_upp - x_low);
      cell(idim2, 0) = ndiv0_;  // looked up on demand (save compute)
      cell(idim2, 1) = idim1;
      cell(idim2, 2) = ig2;
    }  // for iord

    if (!all_cells) return;
    for (S iord = nblocks_; iord < ndim_; ++iord) {
      const S idim2 = order_(iord, 1);
      if (cell(idim2, 0) == ndiv0_) cell(idim2, 0) = diag_cell(idim2, point.x[idim2]);
    }
  }

  /*!
   * @brief Resets the nested grids, sampling order, and accumulated data.
   * @throws std::logic_error if called during an integration iteration.
   */
  void reset() {
    Base::require_not_sampling("reset");
    grid_.fill(T(0));
    /// the diagonal entries (1D grids)
    std::vector<T> flat0(ndiv0_);
    for (S ig0 = 0; ig0 < ndiv0_; ++ig0) {
      flat0[ig0] = T(ig0 + 1) / T(ndiv0_);
    }
    flat0[ndiv0_ - 1] = T(1);  // force
    for (S idim = 0; idim < ndim_; ++idim) {
      std::copy_n(flat0.begin(), ndiv0_, &grid0_(idim, 0));
    }
    /// the off-diagonal conditional grids
    std::vector<T> flat2(ndiv2_);
    for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
      flat2[ig2] = T(ig2 + 1) / T(ndiv2_);
    }
    flat2[ndiv2_ - 1] = T(1);  // force
    for (S idim1 = 0; idim1 < ndim_; ++idim1) {
      for (S idim2 = 0; idim2 < ndim_; ++idim2) {
        if (idim1 == idim2) continue;
        for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
          std::copy_n(flat2.begin(), ndiv2_, &grid_(idim1, idim2, ig1, 0));
        }
      }
    }
    /// initialize the sampling order
    order_.fill(0);
    for (S idim = 0; idim < ndim_; ++idim) {
      order_(idim, 0) = idim;
      order_(idim, 1) = idim;
    }
    nblocks_ = sort_order(grid_, order_, ordered_grid_);
    assert(nblocks_ == ndim_);
    /// also clear the accumulators
    clear_data();
  }

  /*!
   * @brief Adapts the nested grids based on the accumulated data.
   *
   * This method performs the core BASIN update. It refines the diagonal
   * one-dimensional grids and the off-diagonal conditional grids from the
   * accumulated squared weights, then updates the sampling order from the local
   * variance gains above `min_gain`.
   *
   * @throws std::overflow_error if an accumulated grid-cell sum is non-finite.
   *         In that case the grids, sampling order and pending data are unchanged;
   *         call `clear_data()` to discard the invalid batch.
   * @throws std::logic_error if called during an integration iteration.
   */
  void adapt() {
    Base::require_not_sampling("adapt");
    using kakuhen::ndarray::NDArray;
    using kakuhen::ndarray::NDView;

    if (accumulator_count_ <= U(0)) {
      std::cout << "no data collected for adaptation" << std::endl;
      return;
    }

    if (std::ranges::any_of(accumulator_,
                            [](const auto& cell) { return !std::isfinite(cell.value()); })) {
      throw std::overflow_error("adapt: non-finite grid-cell sum");
    }

    const T eps = T(10) * std::numeric_limits<T>::min();
    const T nrm = T(1) / (T(accumulator_count_) * T(accumulator_count_));

    if (opts_.verbosity.value_or(0) > 0) {
      std::cout << "Adapting the grid on " << accumulator_count_ << " collected samples.\n";
    }

    // pre-allocate data structures we need (ndiv0_ = ndiv1_*ndiv2_)
    NDArray<T, S> dval({ndiv0_});
    NDArray<T, S> d({ndiv0_});
    NDArray<T, S> grid_new({ndiv0_});
    T dsum, davg, dacc;
    // weight table
    NDArray<T, S> wgt11({ndiv1_, ndiv1_});
    // merged grids for ndim2_ PDFs
    NDArray<T, S> grid_mrg({ndiv0_});
    S grid_mrg_size;
    // local factor moment E[h^2 w_candidate/w_actual] / N for each block
    NDArray<T, S> moment({ndim_, ndim_});
    moment.fill(T(0));

    for (S idim1 = 0; idim1 < ndim_; ++idim1) {
      ///------------
      /// (1)  adapt the diagonal (idim1==idim2)
      ///------------
      auto d0val = dval.view();
      auto d0 = d.view();
      auto grid0_new = grid_new.view();

      /// initialize `d0val` & check count compatibility
      {
        d0val.fill(T(0));
        U nsum{0};
        for (S ig0 = 0; ig0 < ndiv0_; ++ig0) {
          nsum += accumulator0_(idim1, ig0).count();
          // d = G w with the weight w of this cell (before it is refined)
          d0val(ig0) =
              accumulator0_(idim1, ig0).value() * nrm * (T(ndiv0_) * width(&grid0_(idim1, 0), ig0));
          moment(idim1, idim1) += d0val(ig0);
        }
        assert(nsum == accumulator_count_);
      }

      /// smoothen out and save in `d0`
      {
        d0.fill(T(0));
        dacc = T(0);
        for (S ig0 = 0; ig0 < ndiv0_; ++ig0) {
          if (ig0 == 0) {
            d0(ig0) = (weight_smooth_ + 1) * d0val(ig0) + d0val(ig0 + 1);
          } else if (ig0 == ndiv0_ - 1) {
            d0(ig0) = d0val(ig0 - 1) + (weight_smooth_ + 1) * d0val(ig0);
          } else {
            d0(ig0) = d0val(ig0 - 1) + weight_smooth_ * d0val(ig0) + d0val(ig0 + 1);
          }
          d0(ig0) /= (weight_smooth_ + 2);
          if (d0(ig0) < eps) d0(ig0) = eps;
          dacc += d0(ig0);
        }  // for ig0
      }

      /// dampen (w/o assuming normalization & stable for `eps`)
      dsum = T(0);
      for (S ig0 = 0; ig0 < ndiv0_; ++ig0) {
        if (d0(ig0) > T(0)) {
          d0(ig0) =
              std::pow((T(1) - d0(ig0) / dacc) / (std::log(dacc) - std::log(d0(ig0))), alpha_);
        }
        dsum += d0(ig0);
      }

      /// refine the grid using `d0`
      grid0_new.fill(T(0));
      davg = dsum / T(ndiv0_);
      dacc = T(0);
      S ig0_new = 0;

      if (davg > std::numeric_limits<T>::min()) {
        for (S ig0 = 0; ig0 < ndiv0_; ++ig0) {
          dacc += d0(ig0);
          while (dacc >= davg && ig0_new < ndiv0_) {
            dacc -= davg;
            const T rat = (d0(ig0) > T(0)) ? (dacc / d0(ig0)) : T(0);
            const T safe_rat = std::clamp(rat, T(0), T(1));

            const T x_low = ig0 > 0 ? grid0_(idim1, ig0 - 1) : T(0);
            const T x_upp = grid0_(idim1, ig0);
            grid0_new(ig0_new) = x_low * safe_rat + x_upp * (T(1) - safe_rat);
            ig0_new++;
          }
        }
        while (ig0_new < ndiv0_) {
          grid0_new(ig0_new) = T(1);
          ig0_new++;
        }
      } else {
        std::copy_n(&grid0_(idim1, 0), ndiv0_, grid0_new.data());
      }
      grid0_new(ndiv0_ - 1) = T(1);

      assert(grid0_new(0) >= T(0));
      for (S ig0 = 1; ig0 < ndiv0_; ++ig0)
        assert(grid0_new(ig0) >= grid0_new(ig0 - 1));

      /// compute ig1_new <-> ig1 weight table
      wgt11.fill(T(0));
      for (S ig1_new = 0; ig1_new < ndiv1_; ++ig1_new) {
        const T x1_low_new = ig1_new > 0 ? grid0_new(ig1_new * ndiv2_ - 1) : T(0);
        const T x1_upp_new = grid0_new((ig1_new + 1) * ndiv2_ - 1);
        for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
          const T x1_low = ig1 > 0 ? grid0_(idim1, ig1 * ndiv2_ - 1) : T(0);
          const T x1_upp = grid0_(idim1, (ig1 + 1) * ndiv2_ - 1);
          if (x1_low == x1_upp) continue;  // a rounded zero-width coarse cell has no overlap
          if (x1_low > x1_upp_new) break;
          if (x1_upp < x1_low_new) continue;
          /// new & old have non-vanishing overlap
          const T rat =
              (util::math::min(x1_upp_new, x1_upp) - util::math::max(x1_low_new, x1_low)) /
              (x1_upp - x1_low);
          assert(rat >= T(0) && rat <= T(1));
          wgt11(ig1_new, ig1) = rat;
        }
      }  // for ig1_new

      /// adaptation done & weights saved: overwrite old grid
      for (S ig0 = 0; ig0 < ndiv0_; ++ig0) {
        grid0_(idim1, ig0) = grid0_new(ig0);
      }

      ///------------
      /// (2)  idim2 sub-grid optimization
      ///------------
      for (S idim2 = 0; idim2 < ndim_; ++idim2) {
        if (idim1 == idim2) continue;  // diagonal <-> fine grid already dealt with

        /// set up views into the scratch buffers
        auto d12val = dval.reshape({ndiv1_, ndiv2_});
        auto d12 = d.reshape({ndiv1_, ndiv2_});
        auto grid12_new = grid_new.reshape({ndiv1_, ndiv2_});
        grid12_new.fill(T(0));
        d12val.fill(T(0));
        d12.fill(T(0));

        ///------------
        /// (2.1)  initialize accumulator d-values, smoothen & dampen
        ///------------
        for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
          /// initialize raw sub-grid weights
          const T* row = &grid_(idim1, idim2, ig1, 0);
          for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
            d12val(ig1, ig2) =
                accumulator_(idim1, idim2, ig1, ig2).value() * nrm * (T(ndiv2_) * width(row, ig2));
            moment(idim1, idim2) += d12val(ig1, ig2);
          }  // for ig2

          /// smooth the raw weights
          {
            dacc = T(0);
            for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
              if (ig2 == 0) {
                d12(ig1, ig2) = (weight_smooth_ + 1) * d12val(ig1, ig2) + d12val(ig1, ig2 + 1);
              } else if (ig2 == ndiv2_ - 1) {
                d12(ig1, ig2) = d12val(ig1, ig2 - 1) + (weight_smooth_ + 1) * d12val(ig1, ig2);
              } else {
                d12(ig1, ig2) =
                    d12val(ig1, ig2 - 1) + weight_smooth_ * d12val(ig1, ig2) + d12val(ig1, ig2 + 1);
              }
              d12(ig1, ig2) /= (weight_smooth_ + 2);
              if (d12(ig1, ig2) < eps) d12(ig1, ig2) = eps;
              dacc += d12(ig1, ig2);
            }  // for ig2
          }

          /// damp the smoothed weights without assuming normalization
          for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
            if (d12(ig1, ig2) > T(0)) {
              d12(ig1, ig2) = std::pow(
                  (T(1) - d12(ig1, ig2) / dacc) / (std::log(dacc) - std::log(d12(ig1, ig2))),
                  alpha_);
            }
          }

          /// intentionally no normalization at this stage

        }  // for ig1

        ///------------
        /// (2.2)  apply the weight matrix, accumulate, and adapt the sub-grid
        ///------------
        ///  `d12val` served its purpose; re-use `dval` for super-grid accumulator
        auto d_mrg = dval.reshape({ndiv0_});

        for (S ig1_new = 0; ig1_new < ndiv1_; ++ig1_new) {
          /// (a)  first initialize a merged super-grid
          grid_mrg.fill(T(0));
          grid_mrg_size = 0;
          for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
            if (wgt11(ig1_new, ig1) <= T(0)) continue;
            std::copy_n(&grid_(idim1, idim2, ig1, 0), ndiv2_, grid_mrg.data() + grid_mrg_size);
            grid_mrg_size += ndiv2_;
          }  // for ig1
          std::sort(grid_mrg.data(), grid_mrg.data() + grid_mrg_size);
          /// consistency checks
          if (grid_mrg_size > 0) {
            assert(grid_mrg(0) >= T(0));
            assert(grid_mrg(grid_mrg_size - 1) == T(1));
            for (S ig_mrg = 1; ig_mrg < grid_mrg_size; ++ig_mrg)
              assert(grid_mrg(ig_mrg - 1) <= grid_mrg(ig_mrg));
          }

          /// (b)  apply the overlap weights and accumulate
          d_mrg.fill(T(0));
          dsum = T(0);
          for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
            if (wgt11(ig1_new, ig1) <= T(0)) continue;
            S ig2 = 0;
            T del_x;
            for (S ig_mrg = 0; ig_mrg < grid_mrg_size; ++ig_mrg) {
              if (ig2 >= ndiv2_) break;
              do {
                const T x2_low = ig2 > 0 ? grid_(idim1, idim2, ig1, ig2 - 1) : T(0);
                const T x2_upp = grid_(idim1, idim2, ig1, ig2);
                const T x2_low_mrg = ig_mrg > 0 ? grid_mrg(ig_mrg - 1) : T(0);
                const T x2_upp_mrg = grid_mrg(ig_mrg);
                del_x = util::math::min(x2_upp, x2_upp_mrg) - util::math::max(x2_low, x2_low_mrg);
                if (del_x > T(0)) {
                  const T rat = del_x / (x2_upp - x2_low);
                  assert(rat >= 0 && rat <= 1);
                  const T acc = rat * wgt11(ig1_new, ig1) * d12(ig1, ig2);
                  d_mrg(ig_mrg) += acc;
                  dsum += acc;
                }
                if (grid_(idim1, idim2, ig1, ig2) <= grid_mrg(ig_mrg)) {
                  ig2++;
                  if (ig2 >= ndiv2_) break;
                } else {
                  break;  // -> next `ig_mrg`
                }
              } while (del_x > T(0));
            }  // for ig_mrg
          }  // for ig1

          /// intentionally no normalization (entries can remain near `eps`)
          /// intentionally no extra damping (already applied before the merge)

          /// (c)  refine the sub-grid using `d_mrg` into `ndiv2_` bins
          davg = dsum / T(ndiv2_);
          dacc = T(0);
          S ig2_new = 0;
          if (davg > std::numeric_limits<T>::min()) {
            for (S ig_mrg = 0; ig_mrg < grid_mrg_size; ++ig_mrg) {
              dacc += d_mrg(ig_mrg);
              while (dacc >= davg && ig2_new < ndiv2_) {
                dacc -= davg;
                const T rat = (d_mrg(ig_mrg) > T(0)) ? (dacc / d_mrg(ig_mrg)) : T(0);
                const T safe_rat = std::clamp(rat, T(0), T(1));

                const T x_low = ig_mrg > 0 ? grid_mrg(ig_mrg - 1) : T(0);
                const T x_upp = grid_mrg(ig_mrg);
                grid12_new(ig1_new, ig2_new) = x_low * safe_rat + x_upp * (T(1) - safe_rat);
                ig2_new++;
              }
            }
            while (ig2_new < ndiv2_) {
              grid12_new(ig1_new, ig2_new) = T(1);
              ig2_new++;
            }
          } else {
            // tiny/empty overlaps or underflowed damping leave no usable weights; keep the row
            std::copy_n(&grid_(idim1, idim2, ig1_new, 0), ndiv2_, &grid12_new(ig1_new, 0));
          }
          grid12_new(ig1_new, ndiv2_ - 1) = T(1);

          assert(grid12_new(ig1_new, 0) >= T(0));
          for (S ig2 = 1; ig2 < ndiv2_; ++ig2)
            assert(grid12_new(ig1_new, ig2) >= grid12_new(ig1_new, ig2 - 1));

        }  // for ig1_new

        /// copy back the new grid
        std::copy_n(grid12_new.data(), ndiv0_, &grid_(idim1, idim2, 0, 0));

      }  // for idim2

    }  // for idim1

    ///------------
    /// (3)  apply compatible sampling-order steps ranked by local gain
    ///------------
    update_order(moment);
    nblocks_ = sort_order(grid_, order_, ordered_grid_);

    /// clear the accumulators to prepare for the next iteration
    clear_data();
  }  // adpat

  /*!
   * @brief Clears the pending grid data and their integral statistics.
   *
   * May be called between samples, including from a progress callback. The
   * result returned by the current `integrate()` call still includes all samples.
   */
  void clear_data() {
    accumulator_count_ = U(0);
    std::for_each(accumulator_.begin(), accumulator_.end(), [](auto& acc) { acc.reset(); });
    result_.reset();
  }

  /// @}

  /// @name Utilities
  /// @{

  /*!
   * @brief Prints the current grid structure to standard output.
   *
   * @param prefix A string prefix to prepend to each line of output.
   */
  void print_grid(const std::string& prefix = "") const {
    /// long 1D grid
    for (S idim0 = 0; idim0 < ndim_; ++idim0) {
      std::cout << prefix << "#dim" << idim0 << "\n";
      std::cout << prefix;
      for (S ig0 = 0; ig0 < ndiv0_; ++ig0) {
        std::cout << " " << grid0_(idim0, ig0);
      }
      std::cout << "\n" << prefix << "\n" << prefix << "\n";
    }
    /// joint 2D grid
    for (S idim1 = 0; idim1 < ndim_; ++idim1) {
      for (S idim2 = 0; idim2 < ndim_; ++idim2) {
        if (idim1 == idim2) continue;
        std::cout << prefix << "#dim" << idim1 << idim2 << "\n";
        /// multiple short grids
        for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
          const T x1_min = ig1 > 0 ? grid0_(idim1, ig1 * ndiv2_ - 1) : T(0);
          const T x1_max = grid0_(idim1, (ig1 + 1) * ndiv2_ - 1);
          for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
            const T x2_min = ig2 > 0 ? grid_(idim1, idim2, ig1, ig2 - 1) : T(0);
            const T x2_max = grid_(idim1, idim2, ig1, ig2);
            std::cout << prefix << "  " << x1_min << " " << x1_max;
            std::cout << "  " << x2_min << " " << x2_max << "\n";
          }
          std::cout << prefix << "\n";
        }
        std::cout << prefix << "\n";
      }
    }
  }

  /*!
   * @brief Debug helper to visualize nested grids.
   *
   * @param grid1 The first 1D grid view.
   * @param grid2 The second 1D grid view.
   */
  void nest_grid(const kakuhen::ndarray::NDView<T, S>& grid1,
                 const kakuhen::ndarray::NDView<T, S>& grid2) const {
    assert(grid1.ndim() == 1 && grid2.ndim() == 1);

    S ig1 = 0;
    S ig2 = 0;
    T cdf1 = 0;
    T cdf2 = 0;
    T x = 0;

    //> nested
    while (ig1 < grid1.size() && ig2 < grid2.size()) {
      if (grid1(ig1) < grid2(ig2)) {
        x = grid1(ig1);
        cdf1 = T(ig1 + 1) / T(grid1.size());
        const T x2_low = ig2 > 0 ? grid2(ig2 - 1) : T(0);
        const T x2_upp = grid2(ig2);
        assert(x >= x2_low && x <= x2_upp);
        cdf2 = (ig2 + (x - x2_low) / (x2_upp - x2_low)) / T(grid2.size());
        std::cout << x << "  " << cdf1 << "  " << cdf2 << "\n";

        ig1++;
      } else if (grid1(ig1) > grid2(ig2)) {
        x = grid2(ig2);
        cdf2 = T(ig2 + 1) / T(grid2.size());
        const T x1_low = ig1 > 0 ? grid1(ig1 - 1) : T(0);
        const T x1_upp = grid1(ig1);
        assert(x >= x1_low && x <= x1_upp);
        cdf1 = (ig1 + (x - x1_low) / (x1_upp - x1_low)) / T(grid1.size());
        std::cout << x << "  " << cdf1 << "  " << cdf2 << "\n";

        ig2++;
      } else {
        // equal
        x = grid1(ig1);
        cdf1 = T(ig1 + 1) / T(grid1.size());
        cdf2 = T(ig2 + 1) / T(grid2.size());
        std::cout << x << "  " << cdf1 << "  " << cdf2 << "\n";

        ig1++;
        ig2++;
      }
    }

    //> since both grids end with a `1`, better not have any remainders
    assert(ig1 == grid1.size());
    assert(ig2 == grid2.size());
  }

  /// @}

  /// @name Output & Serialization Implementation
  /// @{

  /*!
   * @brief Prints the internal state of the integrator.
   *
   * @tparam P The printer type.
   * @param prt The printer object.
   */
  template <typename P>
  void print_state(P& prt) const {
    using C = kakuhen::util::printer::Context;
    using namespace kakuhen::util::type;
    prt.print_one("ndiv0", ndiv0_);
    prt.print_one("ndiv1", ndiv1_);
    prt.print_one("ndiv2", ndiv2_);
    /// 1D grid information
    prt.template begin<C::ARRAY>("grid1d");
    prt.break_line();
    {
      for (S idim = 0; idim < ndim_; ++idim) {
        prt.template begin<C::OBJECT>();
        {
          prt.print_array("dims", std::array{idim});
          prt.print_array("grid", &grid0_(idim, 0), ndiv0_, {T(0)});
        }
        prt.template end<C::OBJECT>(true);
      }  // end for idim
    }
    prt.template end<C::ARRAY>(true);
    /// 2D grid information
    prt.template begin<C::ARRAY>("grid2d");
    prt.break_line();
    {
      for (S idim1 = 0; idim1 < ndim_; ++idim1) {
        for (S idim2 = 0; idim2 < ndim_; ++idim2) {
          if (idim1 == idim2) continue;
          prt.template begin<C::OBJECT>();
          {
            prt.print_array("dims", std::array{idim1, idim2});
            prt.template begin<C::ARRAY>("grid");
            for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
              const std::array bin1{ig1 > 0 ? grid0_(idim1, ig1 * ndiv2_ - 1) : T(0),
                                    grid0_(idim1, (ig1 + 1) * ndiv2_ - 1)};
              prt.template begin<C::ARRAY>();
              prt.print_array({}, bin1);
              prt.print_array({}, &grid_(idim1, idim2, ig1, 0), ndiv2_, {T(0)});
              prt.template end<C::ARRAY>();
            }  // end for ig1
            prt.template end<C::ARRAY>();
          }
          prt.template end<C::OBJECT>(true);
        }  // end for idim2
      }  // end for idim1
    }
    prt.template end<C::ARRAY>(true);
    /// dimension sampling information
    prt.template begin<C::ARRAY>("order");
    prt.break_line();
    for (S iord = 0; iord < ndim_; ++iord) {
      prt.print_array({}, &order_(iord, 0), 2);
    }  // end for iord
    prt.template end<C::ARRAY>(true);
  }

  /*!
   * @brief Writes the internal state to a stream.
   * @param out The output stream.
   */
  void write_state_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    serialize_one<S>(out, ndim_);
    serialize_one<S>(out, ndiv1_);
    serialize_one<S>(out, ndiv2_);
    grid_.serialize(out);
    order_.serialize(out);
  }

  /*!
   * @brief Reads the internal state from a stream.
   * @param in The input stream.
   * @throws std::runtime_error if the stream is truncated or corrupt; the
   *         integrator is then left unchanged.
   * @throws std::logic_error if called during an integration iteration.
   */
  void read_state_stream(std::istream& in) {
    Base::require_not_sampling("read_state_stream");
    using namespace kakuhen::util::serialize;
    // read & validate everything before touching any state (the commit below cannot throw)
    S ndim, ndiv1, ndiv2;
    deserialize_one<S>(in, ndim);
    deserialize_one<S>(in, ndiv1);
    deserialize_one<S>(in, ndiv2);
    if (ndim == 0 || ndiv1 < 2 || ndiv2 < 2) {
      throw std::runtime_error("corrupt state (invalid grid dimensions)");
    }
    ndarray::NDArray<T, S> grid;
    grid.deserialize_expected_shape(in, std::array{ndim, ndim, ndiv1, ndiv2});
    ndarray::NDArray<S, S> order;
    order.deserialize_expected_shape(in, std::array{ndim, S(2)});
    const S ndiv0 = ndiv1 * ndiv2;
    for (S idim1 = 0; idim1 < ndim; ++idim1) {
      for (S idim2 = 0; idim2 < ndim; ++idim2) {
        if (idim1 == idim2) {
          detail::validate_grid_row<T>({&grid(idim1, idim2, 0, 0), ndiv0});
        } else {
          for (S ig1 = 0; ig1 < ndiv1; ++ig1) {
            detail::validate_grid_row<T>({&grid(idim1, idim2, ig1, 0), ndiv2});
          }
        }
      }
    }
    ndarray::NDArray<grid_acc_type, S> accumulator({ndim, ndim, ndiv1, ndiv2});
    ndarray::NDArray<T, S> ordered_grid({ndim, ndiv1, ndiv2});
    auto grid0 = grid.reshape({ndim, ndim, ndiv0}).diagonal(0, 1);
    auto accumulator0 = accumulator.reshape({ndim, ndim, ndiv0}).diagonal(0, 1);
    auto ordered_grid0 = ordered_grid.reshape({ndim, ndiv0});
    const S nblocks = sort_order(grid, order, ordered_grid);
    // commit: moves are noexcept and keep the buffers, so the views stay valid
    ndim_ = ndim;
    ndiv1_ = ndiv1;
    ndiv2_ = ndiv2;
    ndiv0_ = ndiv0;
    grid_ = std::move(grid);
    grid0_ = std::move(grid0);
    order_ = std::move(order);
    ordered_grid_ = std::move(ordered_grid);
    ordered_grid0_ = std::move(ordered_grid0);
    accumulator_ = std::move(accumulator);
    accumulator0_ = std::move(accumulator0);
    nblocks_ = nblocks;
    clear_data();
  }

  /*!
   * @brief Writes pending grid data and their cumulative integral statistics to a stream.
   * @param out The output stream.
   */
  void write_data_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    serialize_one<S>(out, ndim_);
    serialize_one<S>(out, ndiv1_);
    serialize_one<S>(out, ndiv2_);
    serialize_one<kakuhen::util::HashValue_t>(out, map_hash());
    result_.serialize(out);
    serialize_one<U>(out, accumulator_count_);
    accumulator_.serialize(out);
  }

  /*!
   * @brief Reads accumulated data from a stream into an integrator without data.
   * @param in The input stream.
   * @throws std::runtime_error if the integrator already holds data, or if the stream is
   *         truncated, corrupt, or incompatible with the current sampling map.
   * @throws std::overflow_error if the sums overflow.
   */
  void read_data_stream(std::istream& in) {
    // a zero adaptation count implies empty grid accumulators
    if (accumulator_count_ != 0 || result_.count() != 0) {
      throw std::runtime_error("integrator already holds data");
    }
    accumulate_data_stream(in);
  }

  /*!
   * @brief Accumulates data from a stream into the current integrator.
   * @param in The input stream.
   * @throws std::runtime_error if the stream is truncated, corrupt, or
   *         incompatible with the current sampling map; grid accumulators must be
   *         finite and non-negative.
   * @throws std::overflow_error if the merged counts or sums overflow.
   * On a throw, the accumulated data is left unchanged.
   */
  void accumulate_data_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    S ndim, ndiv1, ndiv2;
    deserialize_one<S>(in, ndim);
    deserialize_one<S>(in, ndiv1);
    deserialize_one<S>(in, ndiv2);
    if (ndim != ndim_ || ndiv1 != ndiv1_ || ndiv2 != ndiv2_) {
      throw std::runtime_error("incompatible data (grid dimensions mismatch)");
    }
    // every (idim1, idim2) block of ndiv1 * ndiv2 cells holds one entry per adaptation sample
    detail::merge_data_stream(in, map_hash(), result_, accumulator_count_, accumulator_, ndiv0_);
  }

  /// @}

 protected:
  /// integral statistics of the pending data, in step with the grid accumulators
  int_acc_type result_;

  /// division for conditional PDF:  P(x2|x1)
  S ndiv1_;  // number of divisions of the grid along dim 1
  S ndiv2_;  // number of divisions of the grid along dim 2
  S ndiv0_;  // number of divisions of the grid along diagonal
  ndarray::NDArray<T, S> grid_;
  ndarray::NDView<T, S> grid0_;
  /// define the sampling order
  ndarray::NDArray<S, S> order_;
  /// ordered grid
  S nblocks_;
  ndarray::NDArray<T, S> ordered_grid_;
  ndarray::NDView<T, S> ordered_grid0_;

 private:
  /// parameters that controls the grid refinement
  T alpha_{0.75};
  T weight_smooth_{6};  // default of the Vegas implementation
  T min_gain_{T(0.03)};

  U accumulator_count_{0};
  /// per cell sum of G = (f w)^2 / w_b, with w_b the weight of the cell's dimension b as sampled
  ndarray::NDArray<grid_acc_type, S> accumulator_;
  ndarray::NDView<grid_acc_type, S> accumulator0_;

  /// accumulate one sample into the diagonal and all conditional grid cells
  void collect_grid(const point_type& point, const cell_ctx_type& cell, std::span<const T> g) {
    const S nd = ndim_, n2 = ndiv2_, n0 = ndiv0_;
    for (S a = 0; a < nd; ++a) {
      const S ig0 = cell(a, 0);
      accumulator0_(a, ig0).accumulate(g[a]);
      const S ig1 = ig0 / n2;
      //> blocks (a, b, ig1) of all children b are n0 cells apart
      const T* rows = &grid_(a, 0, ig1, 0);
      grid_acc_type* cells = &accumulator_(a, 0, ig1, 0);
      for (S b = 0; b < nd; ++b) {
        if (b == a) continue;
        const std::size_t offset = std::size_t(b) * n0;
        const T* row = rows + offset;
        assert(row == &grid_(a, b, ig1, 0));
        const T x = point.x[b];
        //> active row: reuse the sampled bin, even when x lies on its lower edge
        const S ig2 = cell(b, 1) == a ? cell(b, 2)
                                      : S(util::algorithm::partition_point_index(
                                            row, row + n2, [x](T edge) { return edge < x; }));
        assert(ig2 < n2);
        assert(x >= (ig2 > 0 ? row[ig2 - 1] : T(0)) && x <= row[ig2]);
        assert(cells + offset + ig2 == &accumulator_(a, b, ig1, ig2));
        cells[offset + ig2].accumulate(g[b]);
      }
    }
  }

  /// width of bin `i` of a grid row
  [[nodiscard]] static T width(const T* row, S i) noexcept {
    return row[i] - (i > 0 ? row[i - 1] : T(0));
  }

  [[nodiscard]] T sampled_weight(S b, const cell_ctx_type& cell) const noexcept {
    const S a = cell(b, 1);
    return a == b ? T(ndiv0_) * width(&grid0_(b, 0), cell(b, 0))
                  : T(ndiv2_) * width(&grid_(a, b, cell(a, 0) / ndiv2_, 0), cell(b, 2));
  }

  /// parent of each dimension in the sampling forest (roots are their own parent)
  [[nodiscard]] std::vector<S> parents() const {
    std::vector<S> parent(ndim_);
    for (S i = 0; i < ndim_; ++i)
      parent[order_(i, 1)] = order_(i, 0);
    return parent;
  }

  /*!
   * @brief choose a forest from ranked local factor gains
   *
   * `moment(a, b)` estimates E[h^2 w_ab/w_b] / N under the original sampling map,
   * from all pending batches. Its difference from the active block is b's
   * contribution to a first-order variance model. Reversed pairs use the same raw
   * G, without assuming a becomes a root. Build an acyclic forest from qualifying
   * moves, then restore compatible old parents for unselected dimensions. Keep the
   * best completed forest encountered, including every changed factor's score,
   * rather than treating forced roots as free. Equal gains retain dimension scan
   * order.
   *
   * The model omits interactions between factor changes and uses the grids before
   * refinement, so the chosen order does not guarantee a lower variance.
   */
  void update_order(const ndarray::NDArray<T, S>& moment) {
    //> non-finite samples are excluded from the grids; their batches cannot predict an order.
    if (result_.count() != accumulator_count_ || !result_.is_finite()) return;
    const auto original = parents();
    const T mean = result_.value();
    //> every active block reconstructs the same current second moment: pick 0
    const T var = moment(original[0], 0) - mean * mean / T(accumulator_count_);
    if (!(var > T(0))) return;  // nothing to gain (or no finite estimate)

    //> the *relative* gain in switching the local sampling from
    //> `original[chld] -> chld` to `prnt -> chld`
    const auto local_gain = [&](S prnt, S chld) {
      return (moment(original[chld], chld) - moment(prnt, chld)) / var;
    };

    //> collect all possible moves that give me a positive gain (above threshold)
    struct Move {
      S prnt, chld;
      T gain;
    };
    std::vector<Move> moves;
    for (S chld = 0; chld < ndim_; ++chld) {
      for (S prnt = 0; prnt < ndim_; ++prnt) {
        if (prnt == original[chld]) continue;
        const T gain = local_gain(prnt, chld);
        if (gain > min_gain_) moves.push_back({prnt, chld, gain});
      }
    }
    if (moves.empty()) return;
    std::ranges::stable_sort(moves, std::greater{}, &Move::gain);

    //> function to check whether adding `prnt -> chld` would create a cycle
    //> orderings are forests so cycles are invalid
    const auto creates_cycle = [](std::span<const S> forest, S prnt, S chld) {
      if (prnt == chld) return false;  // diagonal sampling: always ok
      //> if prnt is a descendant of chld, we have a cycle
      S ancestor = prnt;
      while (ancestor != chld && forest[ancestor] != ancestor)
        ancestor = forest[ancestor];
      return ancestor == chld;
    };

    std::vector<S> partial(ndim_);
    std::iota(partial.begin(), partial.end(), S{0});  // start with only roots
    std::vector<bool> chosen(ndim_, false);           // track already picked dimensions
    std::vector<S> candidate(ndim_);
    std::vector<S> best;
    T best_gain = min_gain_;
    for (const auto& move : moves) {
      if (chosen[move.chld] || creates_cycle(partial, move.prnt, move.chld)) continue;
      partial[move.chld] = move.prnt;
      chosen[move.chld] = true;
      candidate = partial;
      //> complete this prefix with old edges in their original root-to-leaf order.
      for (S i = 0; i < ndim_; ++i) {
        const S dim = order_(i, 1), prnt = original[dim];
        if (!chosen[dim] && !creates_cycle(candidate, prnt, dim)) candidate[dim] = prnt;
      }
      T gain = T(0);
      for (S dim = 0; dim < ndim_; ++dim) {
        if (candidate[dim] != original[dim]) gain += local_gain(candidate[dim], dim);
      }
      if (gain > best_gain) {
        best_gain = gain;
        best = candidate;
      }
    }
    if (best.empty()) return;
    for (S dim = 0; dim < ndim_; ++dim) {
      order_(dim, 0) = best[dim];
      order_(dim, 1) = dim;
      if (best[dim] != original[dim] && opts_.verbosity.value_or(0) > 0) {
        const auto sampling =
            best[dim] == dim ? std::string("diagonal") : std::format("given {}", best[dim]);
        std::cout << std::format("Sampling order: dimension {} {}, local variance score {:+.1f}%\n",
                                 dim, sampling, -100 * local_gain(best[dim], dim));
      }
    }
  }

  /// cell `ig0` of `x` in the diagonal grid of `idim`
  [[nodiscard]] S diag_cell(S idim, T x) const {
    const T* row = &grid0_(idim, 0);
    const auto ig0 = static_cast<S>(std::lower_bound(row, row + ndiv0_, x) - row);
    assert(ig0 < ndiv0_ && x >= (ig0 > 0 ? row[ig0 - 1] : T(0)) && x <= row[ig0]);
    return ig0;
  }

  /*!
   * @brief Validates and sorts a sampling order, and builds the matching ordered grid.
   *
   * The order is a forest: every dimension has exactly one parent, and the roots
   * (`prnt == chld`) are the diagonal samplings. They come first, sorted by
   * dimension, followed by the conditional samplings in breadth-first order with
   * children sorted by dimension. All validation completes before the output
   * arrays are modified.
   *
   * @param grid The grid, with shape `{ndim, ndim, ndiv1, ndiv2}`.
   * @param order The sampling order, with shape `{ndim, 2}`; sorted on return.
   * @param ordered_grid The grid rows in sampling order, with shape `{ndim, ndiv1, ndiv2}`.
   * @return The number of diagonal samplings (uncorrelated blocks).
   * @throws std::runtime_error if the order is not a forest over all dimensions.
   */
  static S sort_order(const ndarray::NDArray<T, S>& grid, ndarray::NDArray<S, S>& order,
                      ndarray::NDArray<T, S>& ordered_grid) {
    const S ndim = grid.shape()[0];
    const S ndiv0 = grid.shape()[2] * grid.shape()[3];
    std::vector<S> prnts(ndim, ndim);  // `ndim` marks a missing parent
    for (S i = 0; i < ndim; ++i) {
      const S prnt = order(i, 0), chld = order(i, 1);
      if (prnt >= ndim || chld >= ndim || prnts[chld] != ndim) {
        throw std::runtime_error("invalid sampling order");
      }
      prnts[chld] = prnt;
    }
    ndarray::NDArray<S, S> sorted(order.shape());
    S size = 0;
    const auto push = [&](S prnt, S chld) {
      sorted(size, 0) = prnt;
      sorted(size, 1) = chld;
      ++size;
    };
    for (S dim = 0; dim < ndim; ++dim) {
      if (prnts[dim] == dim) push(dim, dim);
    }
    const S nblocks = size;
    for (S i = 0; i < size; ++i) {  // `size` grows while the trees grow
      const S prnt = sorted(i, 1);
      for (S chld = 0; chld < ndim; ++chld) {
        if (chld != prnt && prnts[chld] == prnt) push(prnt, chld);
      }
    }
    if (size != ndim) {
      throw std::runtime_error("invalid sampling order (cycle)");
    }
    for (S i = 0; i < ndim; ++i) {
      std::copy_n(&grid(sorted(i, 0), sorted(i, 1), 0, 0), ndiv0, &ordered_grid(i, 0, 0));
    }
    order = std::move(sorted);
    return nblocks;
  }
};

}  // namespace kakuhen::integrator
