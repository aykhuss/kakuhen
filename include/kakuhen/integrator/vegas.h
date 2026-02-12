#pragma once

#include "kakuhen/integrator/grid_accumulator.h"
#include "kakuhen/integrator/integrator_base.h"
#include "kakuhen/ndarray/ndarray.h"
#include "kakuhen/util/hash.h"
#include "kakuhen/util/math.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace kakuhen::integrator {

/*!
 * @brief The VEGAS Monte Carlo integrator.
 *
 * This class implements the VEGAS algorithm, a classic method for
 * multi-dimensional Monte Carlo integration based on adaptive importance
 * sampling. It divides the integration space into a grid and adapts the
 * grid over several iterations to concentrate sampling in regions where the
 * integrand is the largest. This leads to a more efficient convergence
 * compared to naive Monte Carlo sampling.
 *
 * The state of the grid can be saved and loaded to resume an integration.
 *
 * @tparam NT The numeric traits for the integrator.
 * @tparam RNG The random number generator to use.
 * @tparam DIST The random number distribution to use.
 */
template <typename NT = util::num_traits_t<>,
          typename RNG = typename IntegratorDefaults<NT>::rng_type,
          typename DIST = typename IntegratorDefaults<NT>::dist_type>
class Vegas : public IntegratorBase<Vegas<NT, RNG, DIST>, NT, RNG, DIST> {
 public:
  static constexpr IntegratorId id = IntegratorId::VEGAS;
  static constexpr IntegratorFeature features =
      IntegratorFeature::STATE | IntegratorFeature::DATA | IntegratorFeature::ADAPT;

  // dependent class: need to explicitly load things from the Base
  using Base = IntegratorBase<Vegas<NT, RNG, DIST>, NT, RNG, DIST>;
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
   * @brief Construct a new Vegas object.
   *
   * @param ndim The number of dimensions of the integration.
   * @param ndiv The number of divisions of the grid along each dimension.
   */
  explicit Vegas(S ndim, S ndiv = 128)
      : Base(ndim),
        ndiv_{ndiv},
        grid_({ndim, ndiv}),
        accumulator_count_{0},
        accumulator_({ndim, ndiv}) {
    assert(ndim > 0 && ndiv > 1);
    reset();
  };

  /*!
   * @brief Construct a new Vegas object by loading state from a file.
   *
   * @param filepath The path to the file containing the integrator state.
   */
  explicit Vegas(const std::filesystem::path& filepath) : Base(0) {
    Base::load(filepath);
  }

  /*!
   * @brief Get the number of grid divisions.
   *
   * @return The value of the number of grid divisions.
   */
  [[nodiscard]] inline S ndiv() const noexcept {
    return ndiv_;
  }

  /*!
   * @brief Set the alpha parameter for grid adaptation.
   *
   * The alpha parameter controls the damping of the grid adaptation. A value
   * of 0 means no damping, while a value greater than 0 will dampen the
   * adaptation. The default value is 0.75.
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
   * @brief Computes a hash of the current grid state.
   *
   * @return A `kakuhen::util::Hash` object representing the state of the grid.
   */
  [[nodiscard]] inline kakuhen::util::Hash hash() const {
    return kakuhen::util::Hash().add(ndim_).add(ndiv_).add(grid_.data(), grid_.size());
  }

  /*!
   * @brief Generates a prefix string for filenames.
   *
   * @param with_hash If true, the hash of the grid is included in the prefix.
   * @return A prefix string for filenames.
   */
  [[nodiscard]] inline std::string prefix(bool with_hash = false) const noexcept {
    std::string pref = "vegas_" + std::to_string(ndim_) + "d";
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

    std::vector<S> grid_vec(ndim_);

    const bool collect_adapt_data = opts_.collect_adapt_data && *opts_.collect_adapt_data;
    for (U i = 0; i < neval; ++i) {
      generate_point(point, grid_vec, i);
      const T fval = point.weight * integrand(point);
      const T fval2 = fval * fval;
      result_.accumulate(fval, fval2);
      if (!collect_adapt_data) continue;
      /// accumulator for the grid
      const T acc = fval2;
      accumulator_count_++;
      for (S idim = 0; idim < ndim_; ++idim) {
        accumulator_(idim, grid_vec[idim]).accumulate(acc);
      }
    }

    return result_;
  }

  /*!
   * @brief Reset the grid to a uniform state.
   *
   * This method resets the grid to a uniform state, where each dimension is
   * divided into `ndiv` equal-sized intervals. It also clears the accumulator
   * and the result.
   */
  void reset() {
    grid_.fill(T(0));
    std::vector<T> flat(ndiv_);
    for (S ig = 0; ig < ndiv_; ++ig) {
      flat[ig] = T(ig + 1) / T(ndiv_);
    }
    for (S idim = 0; idim < ndim_; ++idim) {
      std::copy_n(flat.begin(), ndiv_, &grid_(idim, 0));
    }
    clear_data();
  }

  /*!
   * @brief Adapt the grid based on the accumulated data.
   *
   * This method adapts the grid based on the data that has been accumulated
   * during the integration. It uses the accumulated function values to refine
   * the grid, so that more points are sampled in the regions where the
   * integrand is large.
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

    // pre-allocate data structures
    NDArray<T, S> dval({ndiv_});
    NDArray<T, S> d({ndiv_});
    NDArray<T, S> grid_new({ndiv_});
    T dsum, davg, dacc;

    for (S idim = 0; idim < ndim_; ++idim) {
      /// initialize & check count compatibility
      {
        dval.fill(T(0));
        U nsum{0};
        for (S ig = 0; ig < ndiv_; ++ig) {
          nsum += accumulator_(idim, ig).count();
          dval(ig) = util::math::max(nrm * accumulator_(idim, ig).value(), eps);
        }
        assert(nsum == accumulator_count_);
      }

      /// smoothen out
      d.fill(T(0));
      dacc = T(0);
      for (S ig = 0; ig < ndiv_; ++ig) {
        if (ig == 0) {
          d(ig) = (7 * dval(ig) + dval(ig + 1)) / T(8);
        } else if (ig == ndiv_ - 1) {
          d(ig) = (dval(ig - 1) + 7 * dval(ig)) / T(8);
        } else {
          d(ig) = (dval(ig - 1) + 6 * dval(ig) + dval(ig + 1)) / T(8);
        }
        dacc += d(ig);
      }  // for ig

      /// dampen
      dsum = T(0);
      for (S ig = 0; ig < ndiv_; ++ig) {
        if (d(ig) > T(0)) {
          d(ig) = std::pow((T(1) - d(ig) / dacc) / (std::log(dacc) - std::log(d(ig))), alpha_);
        }
        dsum += d(ig);
      }

      /// refine the grid using `d`
      grid_new.fill(T(0));
      davg = dsum / T(ndiv_);
      dacc = T(0);
      S ig_new = 0;

      // Safety check: if davg is effectively zero, adapting is meaningless/dangerous
      if (davg <= std::numeric_limits<T>::min()) return;

      for (S ig = 0; ig < ndiv_; ++ig) {
        dacc += d(ig);
        // Safety: Check ig_new bounds to prevent buffer overflow
        while (dacc >= davg && ig_new < ndiv_) {
          dacc -= davg;

          // Calculate ratio safely
          const T rat = (d(ig) > T(0)) ? (dacc / d(ig)) : T(0);
          const T safe_rat = std::clamp(rat, T(0), T(1));

          const T x_low = ig > 0 ? grid_(idim, ig - 1) : T(0);
          const T x_upp = grid_(idim, ig);
          grid_new(ig_new) = x_low * safe_rat + x_upp * (T(1) - safe_rat);
          ig_new++;
        }
      }
      // Fill remaining if any (due to FP drift)
      while (ig_new < ndiv_) {
        grid_new(ig_new) = T(1);
        ig_new++;
      }
      grid_new(ndiv_ - 1) = T(1);

      for (S ig = 0; ig < ndiv_; ++ig) {
        grid_(idim, ig) = grid_new[ig];
      }
    }  // for idim

    /// clear the accumulator to prepare for next iteration
    clear_data();
  }

  /*!
   * @brief Clears accumulated integration data.
   */
  void clear_data() {
    accumulator_count_ = U(0);
    std::for_each(accumulator_.begin(), accumulator_.end(), [](auto& acc) { acc.reset(); });
    result_.reset();
  }

  /*!
   * @brief Prints the current grid structure to standard output.
   */
  void print_grid() const {
    for (S i = 0; i < ndim_; ++i) {
      std::cout << "& " << i << "   " << 0. << " ";
      for (S j = 0; j < ndiv_; ++j) {
        std::cout << grid_(i, j) << " ";
      }
      std::cout << "\n";
    }
  }

  /// @}

  /// @name Output & Serialization Implementation
  /// @{

  template <typename P>
  void print_state(P& prt) const {
    using C = kakuhen::util::printer::Context;
    using namespace kakuhen::util::type;
    prt.print_one("ndiv", ndiv_);
    prt.template begin<C::ARRAY>("grid1d");
    prt.break_line();
    {
      std::vector<S> dims(1);
      for (S idim = 0; idim < ndim_; ++idim) {
        dims = {idim};
        prt.template begin<C::OBJECT>();
        {
          prt.print_array("dims", dims);
          prt.print_array("grid", &grid_(idim, 0), ndiv_, {T(0)});
        }
        prt.template end<C::OBJECT>(true);
      }  // end for idim
    }
    prt.template end<C::ARRAY>(true);
  }

  void write_state_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    serialize_one<S>(out, ndim_);
    serialize_one<S>(out, ndiv_);
    grid_.serialize(out);
  }

  void read_state_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    deserialize_one<S>(in, ndim_);
    deserialize_one<S>(in, ndiv_);
    grid_ = ndarray::NDArray<T, S>({ndim_, ndiv_});
    grid_.deserialize(in);
    if (!std::ranges::equal(accumulator_.shape(), grid_.shape())) {
      accumulator_ = ndarray::NDArray<grid_acc_type, S>({ndim_, ndiv_});
    }
    // clear the result & accumulator
    clear_data();
  }

  void write_data_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    serialize_one<S>(out, ndim_);
    serialize_one<S>(out, ndiv_);
    serialize_one<kakuhen::util::HashValue_t>(out, hash().value());
    result_.serialize(out);
    serialize_one<U>(out, accumulator_count_);
    accumulator_.serialize(out);
  }

  void read_data_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    // check that we won't overwrite existing data
    if (accumulator_count_ != 0) {
      throw std::runtime_error("result already has data");
    }
    for (S idim = 0; idim < ndim_; ++idim) {
      for (S ig = 0; ig < ndiv_; ++ig) {
        if (accumulator_(idim, ig).count() != 0) {
          throw std::runtime_error("accumulator already has data");
        }
      }
    }
    // result and accumulator are empty; can just accumulate
    clear_data();  // for good measure
    accumulate_data_stream(in);
  }

  void accumulate_data_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    // read & check for compatibility
    S ndim_chk;
    deserialize_one<S>(in, ndim_chk);
    if (ndim_chk != ndim_) {
      throw std::runtime_error("ndim mismatch");
    }
    S ndiv_chk;
    deserialize_one<S>(in, ndiv_chk);
    if (ndiv_chk != ndiv_) {
      throw std::runtime_error("ndiv mismatch");
    }
    // optimized shape check avoiding vector allocation
    auto g_shape = grid_.shape();
    if (g_shape.size() != 2 || g_shape[0] != ndim_ || g_shape[1] != ndiv_) {
      throw std::runtime_error("grid shape mismatch");
    }
    auto acc_shape = accumulator_.shape();
    if (acc_shape.size() != 2 || acc_shape[0] != ndim_ || acc_shape[1] != ndiv_) {
      throw std::runtime_error("accumulator shape mismatch");
    }

    kakuhen::util::HashValue_t hash_val;
    deserialize_one<kakuhen::util::HashValue_t>(in, hash_val);
    if (hash().value() != hash_val) {
      throw std::runtime_error("hash value mismatch");
    }
    // accumulate result
    int_acc_type result_in;
    result_in.deserialize(in);
    result_.accumulate(result_in);
    // accumulate grid data
    U accumulator_count_in;
    deserialize_one<U>(in, accumulator_count_in);
    accumulator_count_ += accumulator_count_in;
    ndarray::NDArray<grid_acc_type, S> accumulator_in({ndim_, ndiv_});
    accumulator_in.deserialize(in);
    for (S idim = 0; idim < ndim_; ++idim) {
      for (S ig = 0; ig < ndiv_; ++ig) {
        accumulator_(idim, ig).accumulate(accumulator_in(idim, ig));
      }
    }
  }

  /// @}

 private:
  /// parameters that controls the grid refinement
  T alpha_{0.75};

  S ndiv_;  // number of divisions of the grid along each dimension
  ndarray::NDArray<T, S> grid_;
  int_acc_type result_;
  U accumulator_count_{0};
  ndarray::NDArray<grid_acc_type, S> accumulator_;

  /*!
   * @brief Generates a random point in the integration volume.
   *
   * This method uses the current grid state to perform importance sampling.
   *
   * @param point The point object to populate.
   * @param grid_vec A vector to store the grid indices for each dimension.
   * @param sample_index The index of the current sample.
   */
  inline void generate_point(Point<num_traits>& point, std::vector<S>& grid_vec,
                             U sample_index = U(0)) {
    point.sample_index = sample_index;
    point.weight = T(1);
    for (S idim = 0; idim < ndim_; ++idim) {
      T rand = Base::ran();
      // intervals rand in [ i/ndiv_ , (i+1)/ndiv_ ] mapped to i
      const S ig = S(rand * ndiv_);  // 0 .. (ndiv_-1)
      assert(ig >= 0 && ig < ndiv_);
      assert(rand * ndiv_ >= T(ig) && rand * ndiv_ <= T(ig + 1));
      // map rand back to [ 0, 1 ]
      rand = rand * ndiv_ - T(ig);
      assert(rand >= T(0) && rand <= T(1));
      const T x_low = ig > 0 ? grid_(idim, ig - 1) : T(0);
      const T x_upp = grid_(idim, ig);
      point.x[idim] = x_low + rand * (x_upp - x_low);
      // point.x[idim] = x_low * (T(1) - rand) + x_upp * rand;
      grid_vec[idim] = ig;
      point.weight *= ndiv_ * (x_upp - x_low);
    }
  }

};  // class Vegas

}  // namespace kakuhen::integrator
