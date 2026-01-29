#pragma once

#include "kakuhen/integrator/grid_accumulator.h"
#include "kakuhen/integrator/integrator_base.h"
#include "kakuhen/ndarray/ndarray.h"
#include "kakuhen/ndarray/ndview.h"
#include "kakuhen/util/hash.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace kakuhen::integrator {

/*!
 * @brief An integrator based on Blockwise Adaptive Sampling with Interdimensional Nesting (BASIN).
 *
 * This class implements a sophisticated adaptive Monte Carlo integration algorithm
 * that models correlations between integration dimensions. It uses a nested
 * grid structure where each dimension's grid is conditioned on the value of
 * another, allowing it to adapt to complex integrand shapes.
 *
 * The algorithm first adapts one-dimensional marginal grids and then uses the
 * Earth Mover's Distance (EMD) to score the correlations between dimensions,
 * determining an optimal sampling order for subsequent iterations. This can
 * significantly improve efficiency for integrands with strong inter-dimensional
 * dependencies.
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
  static constexpr IntegratorId id = IntegratorId::BASIN;
  static constexpr IntegratorFeature features =
      IntegratorFeature::STATE | IntegratorFeature::DATA | IntegratorFeature::ADAPT;

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
        ndiv0_{ndiv1 * ndiv2},
        grid_({ndim, ndim, ndiv1, ndiv2}),
        accumulator_count_{0},
        accumulator_({ndim, ndim, ndiv1, ndiv2}),
        order_({ndim, 2}) {
    assert(ndim > 0 && ndiv1 > 1 && ndiv2 > 1);
    grid0_ = grid_.reshape({ndim_, ndim_, ndiv0_}).diagonal(0, 1);
    accumulator0_ = accumulator_.reshape({ndim_, ndim_, ndiv0_}).diagonal(0, 1);
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
   * @brief Set the alpha parameter for grid adaptation.
   *
   * @param alpha The new value for the alpha parameter.
   */
  inline void set_alpha(const T& alpha) noexcept {
    assert(alpha >= T(0));
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
   * @param weight_smooth The new value for the weight smoothing parameter.
   */
  inline void set_weight_smooth(const T& weight_smooth) noexcept {
    assert(weight_smooth >= T(1));
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
   * @brief Set the minimum score for dimension correlation.
   *
   * @param min_score The new value for the minimum score.
   */
  inline void set_min_score(const T& min_score) noexcept {
    assert((min_score >= T(0)) && (min_score < T(1)));
    min_score_ = min_score;
  }
  /*!
   * @brief Get the minimum score for dimension correlation.
   *
   * @return The current value of the minimum score.
   */
  [[nodiscard]] inline T min_score() const noexcept {
    return min_score_;
  }

  /*!
   * @brief Computes a hash of the current grid state.
   *
   * @return A `kakuhen::util::Hash` object representing the state of the grid.
   */
  [[nodiscard]] inline kakuhen::util::Hash hash() const {
    return kakuhen::util::Hash().add(ndim_).add(ndiv1_).add(ndiv2_).add(grid_.data(), grid_.size());
  }

  /*!
   * @brief Generates a prefix string for filenames.
   *
   * @param with_hash If true, the hash of the grid is included in the prefix.
   * @return A prefix string for filenames.
   */
  [[nodiscard]] inline std::string prefix(bool with_hash = false) const noexcept {
    std::string pref = "basin_" + std::to_string(ndim_) + "d";
    if (with_hash) pref += "_" + hash().encode_hex();
    return pref;
  }

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
  int_acc_type integrate_impl(I&& integrand, U neval) {
    result_.reset();

    Point<num_traits> point{ndim_, opts_.user_data.value_or(nullptr)};

    std::vector<S> grid_vec(ndim_);  // vetor in `ndiv0_` space

    for (U i = 0; i < neval; ++i) {
      generate_point(point, grid_vec, i);
      const T fval = point.weight * integrand(point);
      const T fval2 = fval * fval;
      result_.accumulate(fval, fval2);
      /// accumulators for the grid
      const T acc = fval2;
      accumulator_count_++;
      for (S idim = 0; idim < ndim_; ++idim) {
        const S ig0 = grid_vec[idim];
        accumulator0_(idim, ig0).accumulate(acc);
        const S ig1 = ig0 / ndiv2_;
        for (S idim2 = 0; idim2 < ndim_; ++idim2) {
          if (idim2 == idim) continue;
          S ig2 = 0;
          S ig2_hi = ndiv2_;
          while (ig2 < ig2_hi) {
            const S mid = ig2 + ((ig2_hi - ig2) >> 1);
            if (point.x[idim2] < grid_(idim, idim2, ig1, mid))
              ig2_hi = mid;
            else
              ig2 = mid + 1;
          }
          assert(ig2 >= 0 && ig2 < ndiv2_);
          assert(point.x[idim2] >= (ig2 > 0 ? grid_(idim, idim2, ig1, ig2 - 1) : T(0)));
          assert(point.x[idim2] <= grid_(idim, idim2, ig1, ig2));
          accumulator_(idim, idim2, ig1, ig2).accumulate(acc);
        }
      }
    }

    return result_;
  }

  /*!
   * @brief Resets the grid and all accumulators to their initial state.
   */
  void reset() {
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
    /// the non-diagonal entries (2D grids)
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
    /// also cear the accumulators
    clear_data();
  }

  /*!
   * @brief Adapts the nested grids based on the accumulated data.
   *
   * This method performs the core logic of the BASIN algorithm. It refines
   * the diagonal (1D) and off-diagonal (2D) grids based on the variance
   * of the integrand and then determines the optimal sampling order for the
   * next iteration based on dimension correlations.
   */
  void adapt() {
    using kakuhen::ndarray::NDArray;
    using kakuhen::ndarray::NDView;

    if (accumulator_count_ <= U(0)) {
      std::cout << "no data collected for adaption" << std::endl;
      return;
    }

    const T eps = T(10) * std::numeric_limits<T>::min();
    const T nrm = T(1) / (T(accumulator_count_) * T(accumulator_count_));

    if (opts_.verbosity && *opts_.verbosity > 0) {
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
          d0val(ig0) = nrm * accumulator0_(idim1, ig0).value();
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
      }
      grid0_new(ndiv0_ - 1) = T(1);

      assert(grid0_new(0) > T(0));
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
          if (x1_low > x1_upp_new) break;
          if (x1_upp < x1_low_new) continue;
          /// new & old have non-vanishing overlap
          const T rat =
              (std::min(x1_upp_new, x1_upp) - std::max(x1_low_new, x1_low)) / (x1_upp - x1_low);
          assert(rat >= 0. && rat <= 1.);
          wgt11(ig1_new, ig1) = rat;
        }
      }  // for ig1_new

      /// adation done & weights saved: overwrite old grid
      for (S ig0 = 0; ig0 < ndiv0_; ++ig0) {
        grid0_(idim1, ig0) = grid0_new(ig0);
      }

      ///------------
      /// (2)  idim2 sub-grid optimization
      ///------------
      for (S idim2 = 0; idim2 < ndim_; ++idim2) {
        if (idim1 == idim2) continue;  // diagonal <-> fine grid already dealt with

        /// set up views
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
          /// init dval
          for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
            d12val(ig1, ig2) = nrm * accumulator_(idim1, idim2, ig1, ig2).value();
          }  // for ig2

          /// smoothen
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

          /// dampen (w/o assuming normalization & stable for `eps`)
          for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
            if (d12(ig1, ig2) > T(0)) {
              d12(ig1, ig2) = std::pow(
                  (T(1) - d12(ig1, ig2) / dacc) / (std::log(dacc) - std::log(d12(ig1, ig2))),
                  alpha_);
            }
          }

          /// attention:  no normalization!

        }  // for ig1

        ///------------
        /// (2.2)  apply the weight matrix; accumulate & adapt sub-grid
        ///------------
        ///  `d12val` served its purpose; re-use `dval` for super-grid accumulator
        auto d_mrg = dval.reshape({ndiv0_});

        for (S ig1_new = 0; ig1_new < ndiv1_; ++ig1_new) {
          /// (a)  first need to initialize a super-grid
          grid_mrg.fill(T(0));
          grid_mrg_size = 0;
          for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
            if (wgt11(ig1_new, ig1) <= T(0)) continue;
            std::copy_n(&grid_(idim1, idim2, ig1, 0), ndiv2_, grid_mrg.data() + grid_mrg_size);
            grid_mrg_size += ndiv2_;
          }  // for ig1
          std::sort(grid_mrg.data(), grid_mrg.data() + grid_mrg_size);
          /// checks
          if (grid_mrg_size > 0) {
            assert(grid_mrg(0) > T(0));
            assert(grid_mrg(grid_mrg_size - 1) == T(1));
            for (S ig_mrg = 1; ig_mrg < grid_mrg_size; ++ig_mrg)
              assert(grid_mrg(ig_mrg - 1) <= grid_mrg(ig_mrg));
          }

          /// (b)  apply weights and accumulate
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
                del_x = std::min(x2_upp, x2_upp_mrg) - std::max(x2_low, x2_low_mrg);
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

          /// attention: no normalization (potential ~eps entries)
          /// attention: no damping (already done at the pre-merge step)

          /// (c)  refine the sub-grid using `d_mrg` -> into ndiv2_ bins
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
          }
          grid12_new(ig1_new, ndiv2_ - 1) = T(1);

          assert(grid12_new(ig1_new, 0) > T(0));
          for (S ig2 = 1; ig2 < ndiv2_; ++ig2)
            assert(grid12_new(ig1_new, ig2) >= grid12_new(ig1_new, ig2 - 1));

        }  // for ig1_new

        /// copy back the new grid
        std::copy_n(grid12_new.data(), ndiv0_, &grid_(idim1, idim2, 0, 0));

      }  // for idim2

    }  // for idim1

    ///------------
    /// (3)  determine the sampling order
    ///------------
    NDArray<T, S> scores({ndim_, ndim_});

    /// (a)  find the scores for each pair of dimensions
    scores.fill(T(0));
    for (S idim1 = 0; idim1 < ndim_; ++idim1) {
      scores(idim1, idim1) = T(1);  // positive number to signal that this dimension is still active
      for (S idim2 = 0; idim2 < ndim_; ++idim2) {
        if (idim1 == idim2) continue;
        for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
          scores(idim1, idim2) += emd(grid_.slice({{idim2}, {idim2}, {}, {}}).reshape({ndiv0_}),
                                      grid_.slice({{idim1}, {idim2}, {ig1}, {}}).reshape({ndiv2_}));
        }
        scores(idim1, idim2) /= T(ndiv1_);
      }  // for idim2
    }  // for idim1

    /// (b)  determine the order from the scores
    for (S iord = 0; iord < ndim_; ++iord) {
      T max_score = T(-1);
      S max_idim1 = ndim_;  // invalid value for init (-1: not allowed as unsigned)
      S max_idim2 = ndim_;

      /// find highest avg score
      for (S idim1 = 0; idim1 < ndim_; ++idim1) {
        if (scores(idim1, idim1) <= T(0)) continue;  // already selected
        T avg_score = T(0);
        S count = 0;
        for (S idim2 = 0; idim2 < ndim_; ++idim2) {
          if (idim1 == idim2) continue;
          if (scores(idim1, idim2) <= T(0)) continue;
          avg_score += scores(idim1, idim2);
          count++;
        }
        /// we need to penalize sampling of new dimensions
        if (count > 0) avg_score /= pentalty_fac_score_ * T(count);
        if (avg_score > max_score) {
          max_score = avg_score;
          max_idim1 = idim1;
          max_idim2 = idim1;
        }
      }  // for idim1

      /// find highest score w.r.t. already sampled dimensions
      for (S ichk = 0; ichk < iord; ++ichk) {
        const S idim1 = order_(ichk, 1);  // idim1 is an idim2 of a previous step
        for (S idim2 = 0; idim2 < ndim_; ++idim2) {
          if (idim1 == idim2) continue;
          if (scores(idim1, idim2) < min_score_) {
            continue;
          }
          if (scores(idim1, idim2) > max_score) {
            max_score = scores(idim1, idim2);
            max_idim1 = idim1;
            max_idim2 = idim2;
          }
        }
      }

      /// register the order and invalidate the scores
      order_(iord, 0) = max_idim1;
      order_(iord, 1) = max_idim2;
      scores(max_idim2, max_idim2) = T(-1);
      for (S idim = 0; idim < ndim_; ++idim) {
        scores(idim, max_idim2) = T(-1);
      }

    }  // for iord

#ifndef NDEBUG
    /// check order that all dimensions are covered
    bool all_dimensions_covered = true;
    for (S idim = 0; idim < ndim_; ++idim) {
      bool covered = false;
      for (S iord = 0; iord < ndim_; ++iord) {
        if (order_(iord, 1) == idim) {
          covered = true;
          break;
        }
      }
      if (!covered) {
        all_dimensions_covered = false;
        std::cout << "Dimension " << idim << " not covered in order\n";
      }
    }
    assert(all_dimensions_covered);
#endif
    /// clear the accumulator to prepare for next iteration
    clear_data();
  }  // adpat

  /*!
   * @brief Clears accumulated integration data.
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
      std::vector<S> dims(1);
      for (S idim = 0; idim < ndim_; ++idim) {
        dims = {idim};
        prt.template begin<C::OBJECT>();
        {
          prt.print_array("dims", dims);
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
      std::vector<S> dims(2);
      std::vector<T> bin1(2);
      for (S idim1 = 0; idim1 < ndim_; ++idim1) {
        for (S idim2 = 0; idim2 < ndim_; ++idim2) {
          if (idim1 == idim2) continue;
          dims = {idim1, idim2};
          prt.template begin<C::OBJECT>();
          {
            prt.print_array("dims", dims);
            prt.template begin<C::ARRAY>("grid");
            for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
              bin1.at(0) = ig1 > 0 ? grid0_(idim1, ig1 * ndiv2_ - 1) : T(0);
              bin1.at(1) = grid0_(idim1, (ig1 + 1) * ndiv2_ - 1);
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
    {
      std::vector<S> dep(2);
      for (S iord = 0; iord < ndim_; ++iord) {
        dep = {order_(iord, 0), order_(iord, 1)};
        prt.print_array({}, &order_(iord, 0), 2);
      }  // end for iord
    }
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
   */
  void read_state_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    deserialize_one<S>(in, ndim_);
    deserialize_one<S>(in, ndiv1_);
    deserialize_one<S>(in, ndiv2_);
    ndiv0_ = ndiv1_ * ndiv2_;
    grid_ = ndarray::NDArray<T, S>({ndim_, ndim_, ndiv1_, ndiv2_});
    grid_.deserialize(in);
    grid0_ = grid_.reshape({ndim_, ndim_, ndiv0_}).diagonal(0, 1);
    if (!std::ranges::equal(accumulator_.shape(), grid_.shape()) ||
        !std::ranges::equal(accumulator0_.shape(), grid0_.shape())) {
      accumulator_ = ndarray::NDArray<grid_acc_type, S>({ndim_, ndim_, ndiv1_, ndiv2_});
      accumulator0_ = accumulator_.reshape({ndim_, ndim_, ndiv0_}).diagonal(0, 1);
    }
    order_ = ndarray::NDArray<S, S>({ndim_, 2});
    order_.deserialize(in);
    /// reset the result & accumulator
    clear_data();
  }

  /*!
   * @brief Writes accumulated data to a stream.
   * @param out The output stream.
   */
  void write_data_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    serialize_one<S>(out, ndim_);
    serialize_one<S>(out, ndiv1_);
    serialize_one<S>(out, ndiv2_);
    serialize_one<kakuhen::util::HashValue_t>(out, hash().value());
    result_.serialize(out);
    serialize_one<U>(out, accumulator_count_);
    accumulator_.serialize(out);
  }

  /*!
   * @brief Reads accumulated data from a stream.
   * @param in The input stream.
   */
  void read_data_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    //> check that we won't overwrite existing data
    if (accumulator_count_ != 0) {
      throw std::runtime_error("result already has data");
    }
    for (S idim1 = 0; idim1 < ndim_; ++idim1) {
      for (S idim2 = 0; idim2 < ndim_; ++idim2) {
        for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
          for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
            if (accumulator_(idim1, idim2, ig1, ig2).count() != 0) {
              throw std::runtime_error("accumulator already has data");
            }
          }
        }
      }
    }
    //> result and accumulator are empty; can just accumulate
    clear_data();  // for good measure
    accumulate_data_stream(in);
  }

  /*!
   * @brief Accumulates data from a stream into the current integrator.
   * @param in The input stream.
   */
  void accumulate_data_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    //> read & check for compatibility
    S ndim_chk;
    deserialize_one<S>(in, ndim_chk);
    if (ndim_chk != ndim_) {
      throw std::runtime_error("ndim mismatch");
    }
    S ndiv1_chk;
    deserialize_one<S>(in, ndiv1_chk);
    if (ndiv1_chk != ndiv1_) {
      throw std::runtime_error("ndiv1 mismatch");
    }
    S ndiv2_chk;
    deserialize_one<S>(in, ndiv2_chk);
    if (ndiv2_chk != ndiv2_) {
      throw std::runtime_error("ndiv2 mismatch");
    }

    // Optimized shape checks
    auto grid_sh = grid_.shape();
    if (grid_sh.size() != 4 || grid_sh[0] != ndim_ || grid_sh[1] != ndim_ || grid_sh[2] != ndiv1_ ||
        grid_sh[3] != ndiv2_) {
      throw std::runtime_error("grid shape mismatch");
    }
    auto grid0_sh = grid0_.shape();
    if (grid0_sh.size() != 2 || grid0_sh[0] != ndim_ || grid0_sh[1] != ndiv0_) {
      throw std::runtime_error("grid0 shape mismatch");
    }
    auto acc_sh = accumulator_.shape();
    if (acc_sh.size() != 4 || acc_sh[0] != ndim_ || acc_sh[1] != ndim_ || acc_sh[2] != ndiv1_ ||
        acc_sh[3] != ndiv2_) {
      throw std::runtime_error("accumulator shape mismatch");
    }
    auto acc0_sh = accumulator0_.shape();
    if (acc0_sh.size() != 2 || acc0_sh[0] != ndim_ || acc0_sh[1] != ndiv0_) {
      throw std::runtime_error("accumulator0 shape mismatch");
    }
    auto order_sh = order_.shape();
    if (order_sh.size() != 2 || order_sh[0] != ndim_ || order_sh[1] != 2) {
      throw std::runtime_error("order shape mismatch");
    }

    kakuhen::util::HashValue_t hash_val;
    deserialize_one<kakuhen::util::HashValue_t>(in, hash_val);
    if (hash().value() != hash_val) {
      throw std::runtime_error("hash value mismatch");
    }
    //> accumulate result
    int_acc_type result_in;
    result_in.deserialize(in);
    result_.accumulate(result_in);
    //> accumulate grid data
    U accumulator_count_in;
    deserialize_one<U>(in, accumulator_count_in);
    accumulator_count_ += accumulator_count_in;
    ndarray::NDArray<grid_acc_type, S> accumulator_in({ndim_, ndim_, ndiv1_, ndiv2_});
    accumulator_in.deserialize(in);
    for (S idim1 = 0; idim1 < ndim_; ++idim1) {
      for (S idim2 = 0; idim2 < ndim_; ++idim2) {
        for (S ig1 = 0; ig1 < ndiv1_; ++ig1) {
          for (S ig2 = 0; ig2 < ndiv2_; ++ig2) {
            accumulator_(idim1, idim2, ig1, ig2).accumulate(accumulator_in(idim1, idim2, ig1, ig2));
          }
        }
      }
    }
  }

  /// @}

 private:
  /// parameters that controls the grid refinement
  T alpha_{0.75};
  T weight_smooth_{3};
  T min_score_{0.05};
  T pentalty_fac_score_{2};

  /// division for conditional PDF:  P(x2|x1)
  S ndiv1_;  // number of divisions of the grid along dim 1
  S ndiv2_;  // number of divisions of the grid along dim 2
  S ndiv0_;  // number of divisions of the grid along diagonal
  ndarray::NDArray<T, S> grid_;
  ndarray::NDView<T, S> grid0_;
  int_acc_type result_;
  U accumulator_count_{0};
  ndarray::NDArray<grid_acc_type, S> accumulator_;
  ndarray::NDView<grid_acc_type, S> accumulator0_;
  /// define the sampling order
  ndarray::NDArray<S, S> order_;

  inline void generate_point(Point<num_traits>& point, std::vector<S>& grid_vec,
                             U sample_index = U(0)) {
    point.sample_index = sample_index;
    point.weight = T(1);
    for (S iord = 0; iord < ndim_; ++iord) {
      T rand = Base::ran();
      if (order_(iord, 0) == order_(iord, 1)) {
        /// (a) diagnoal map
        const S idim0 = order_(iord, 0);
        //> intervals rand in [ i/ndiv0_ , (i+1)/ndiv0_ ] mapped to i
        const S ig0 = S(rand * ndiv0_);  // 0 .. (ndiv0_-1)
        assert(ig0 >= 0 && ig0 < ndiv0_);
        assert(rand * ndiv0_ >= T(ig0) && rand * ndiv0_ <= T(ig0 + 1));
        //> map rand back to [ 0, 1 ]
        rand = rand * ndiv0_ - T(ig0);
        assert(rand >= T(0) && rand <= T(1));
        const T x_low = ig0 > 0 ? grid0_(idim0, ig0 - 1) : T(0);
        const T x_upp = grid0_(idim0, ig0);
        point.x[idim0] = x_low + rand * (x_upp - x_low);
        // point.x[idim0] = x_low * (T(1) - rand) + x_upp * rand;
        point.weight *= ndiv0_ * (x_upp - x_low);
        grid_vec[idim0] = ig0;

      } else {
        /// (b) conditional map
        const S idim1 = order_(iord, 0);
        const S idim2 = order_(iord, 1);
        /// check that the 1st dimension is set properly
        assert(point.x[idim1] >= T(0) && point.x[idim1] <= T(1));
        assert(grid_vec[idim1] >= 0 && grid_vec[idim1] < ndiv0_);
        /// this is correct because `grid_vec` always stores `ig0`
        const S ig1 = grid_vec[idim1] / ndiv2_;
        assert(ig1 >= 0 && ig1 < ndiv1_);
        //> intervals rand in [ i/ndiv2_ , (i+1)/ndiv2_ ] mapped to i
        const S ig2 = S(rand * ndiv2_);  // 0 .. (ndiv2_-1)
        assert(ig2 >= 0 && ig2 < ndiv2_);
        assert(rand * ndiv2_ >= T(ig2) && rand * ndiv2_ <= T(ig2 + 1));
        //> map rand back to [ 0, 1 ]
        rand = rand * ndiv2_ - T(ig2);
        assert(rand >= T(0) && rand <= T(1));
        const T x_low = ig2 > 0 ? grid_(idim1, idim2, ig1, ig2 - 1) : T(0);
        const T x_upp = grid_(idim1, idim2, ig1, ig2);
        const T x = x_low + rand * (x_upp - x_low);
        point.x[idim2] = x;
        // point.x[idim2] = x_low * (T(1) - rand) + x_upp * rand;
        point.weight *= ndiv2_ * (x_upp - x_low);
        /// need to get index ig0 for idim2
        S ig0 = 0;
        S ig0_hi = ndiv0_;
        while (ig0 < ig0_hi) {
          const S mid =
              ig0 + ((ig0_hi - ig0) >> 1);  // same as `(ig0 + ig0_hi)/2` but safer against overflow
          if (x < grid0_(idim2, mid))
            ig0_hi = mid;
          else
            ig0 = mid + 1;
        }
        assert(ig0 >= 0 && ig0 < ndiv0_);
        assert(x >= (ig0 > 0 ? grid0_(idim2, ig0 - 1) : 0) && x <= grid0_(idim2, ig0));
        grid_vec[idim2] = ig0;
      }
    }
  }

  /*!
   * @brief Computes the Earth Mover's Distance (EMD) between two 1D grids.
   *
   * This function calculates the Wasserstein-1 distance between two probability
   * distributions represented by their cumulative grid boundaries. The grids
   * are assumed to span [0, 1] and be monotonically increasing.
   *
   * @param grid1 The first grid view (must be 1D).
   * @param grid2 The second grid view (must be 1D).
   * @return The Earth Mover's Distance between the two distributions.
   */
  [[nodiscard]] T emd(const kakuhen::ndarray::NDView<T, S>& grid1,
                      const kakuhen::ndarray::NDView<T, S>& grid2) {
    assert(grid1.ndim() == 1 && grid2.ndim() == 1);

    T emd_val = 0;

    S ig1 = 0;
    S ig2 = 0;
    T cdf1 = 0;
    T cdf2 = 0;
    T x = 0;

    //> nested
    while (ig1 < grid1.size() && ig2 < grid2.size()) {
      assert(ig1 > 0 ? grid1(ig1) >= grid1(ig1 - 1) : grid1(ig1) >= 0);
      assert(ig2 > 0 ? grid2(ig2) >= grid2(ig2 - 1) : grid2(ig2) >= 0);

      T x_nxt = 0;
      T cdf1_nxt = 0;
      T cdf2_nxt = 0;
      bool advance_ig1 = false;
      bool advance_ig2 = false;

      if (grid1(ig1) < grid2(ig2)) {
        x_nxt = grid1(ig1);
        cdf1_nxt = T(ig1 + 1) / T(grid1.size());
        const T x2_low = ig2 > 0 ? grid2(ig2 - 1) : T(0);
        const T x2_upp = grid2(ig2);
        assert(x_nxt >= x2_low && x_nxt <= x2_upp);
        cdf2_nxt = (ig2 + (x_nxt - x2_low) / (x2_upp - x2_low)) / T(grid2.size());
        advance_ig1 = true;
      } else if (grid1(ig1) > grid2(ig2)) {
        x_nxt = grid2(ig2);
        cdf2_nxt = T(ig2 + 1) / T(grid2.size());
        const T x1_low = ig1 > 0 ? grid1(ig1 - 1) : T(0);
        const T x1_upp = grid1(ig1);
        assert(x_nxt >= x1_low && x_nxt <= x1_upp);
        cdf1_nxt = (ig1 + (x_nxt - x1_low) / (x1_upp - x1_low)) / T(grid1.size());
        advance_ig2 = true;
      } else {
        // equal
        x_nxt = grid1(ig1);
        cdf1_nxt = T(ig1 + 1) / T(grid1.size());
        cdf2_nxt = T(ig2 + 1) / T(grid2.size());
        advance_ig1 = true;
        advance_ig2 = true;
      }
      assert(x_nxt >= x);

      //> accumulate EMD; need to check for sign flips
      const T dcdf = cdf1 - cdf2;
      const T dcdf_nxt = cdf1_nxt - cdf2_nxt;
      bool sign_flip = dcdf * dcdf_nxt < T(0);
      if (!sign_flip) {
        /// no sign flip; just accumulate
        emd_val += 0.5 * std::abs(dcdf + dcdf_nxt) * (x_nxt - x);
      } else {
        /// sign flip; need to find the crossing point
        const T x_cross = (dcdf * x_nxt - dcdf_nxt * x) / (dcdf - dcdf_nxt);
        assert(x_cross >= x && x_cross <= x_nxt);
        // accumulate two triangles
        emd_val +=
            0.5 * std::abs(dcdf) * (x_cross - x) + 0.5 * std::abs(dcdf_nxt) * (x_nxt - x_cross);
      }

      x = x_nxt;
      cdf1 = cdf1_nxt;
      cdf2 = cdf2_nxt;
      if (advance_ig1) ig1++;
      if (advance_ig2) ig2++;
    }  // while

    //> since both grids must end with a `1`, better not have any remainders
    assert(ig1 == grid1.size() && grid1(ig1 - 1) == T(1));
    assert(ig2 == grid2.size() && grid2(ig2 - 1) == T(1));

    return emd_val;
  }
};

}  // namespace kakuhen::integrator
