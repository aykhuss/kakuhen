// Vegas - the VEGAS algorithm based on adaptive importance sampling
#pragma once

#include "kakuhen/integrator/grid_accumulator.h"
#include "kakuhen/integrator/integrator_base.h"
#include "kakuhen/integrator/numeric_traits.h"
#include "kakuhen/integrator/point.h"
#include "kakuhen/ndarray/ndarray.h"
#include "kakuhen/util/hash.h"
#include "kakuhen/util/serialize.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace kakuhen::integrator {

template <typename NT = num_traits_t<>, typename RNG = typename IntegratorDefaults<NT>::rng_type,
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
  // using typename Base::result_type;
  using grid_acc_type = GridAccumulator<value_type, count_type>;
  //  member variables
  using Base::ndim_;
  using Base::opts_;

  explicit Vegas(size_type ndim, size_type ndiv = 256)
      : Base(ndim),
        alpha_{1.2},
        nmax_smooth_{3},
        ndiv_{ndiv},
        grid_({ndim, ndiv}),
        accumulator_({ndim, ndiv}) {
    assert(ndim > 0 && ndiv > 0);
    reset();
  };

  explicit Vegas(const std::filesystem::path& filepath) : Base(0) {
    Base::load(filepath);
  }

  inline void set_alpha(value_type alpha) noexcept {
    assert(alpha >= value_type(0));
    alpha_ = alpha;
  }
  value_type alpha() const noexcept {
    return alpha_;
  }

  inline void set_nmax_smooth(size_type nmax_smooth) noexcept {
    nmax_smooth_ = nmax_smooth;
  }
  size_type nmax_smooth() const noexcept {
    return nmax_smooth_;
  }

  inline kakuhen::util::Hash hash() const {
    return kakuhen::util::Hash().add(ndim_).add(ndiv_).add(grid_.data(), grid_.size());
  }

  inline std::string prefix(bool with_hash = false) const noexcept {
    std::string pref = "vegas_" + std::to_string(ndim_) + "d";
    if (with_hash) pref += "_" + hash().encode_hex();
    return pref;
  }

  template <typename I>
  int_acc_type integrate_impl(I&& integrand, count_type neval) {
    result_.reset();

    Point<num_traits> point{ndim_, opts_.user_data.value_or(nullptr)};

    std::vector<size_type> grid_vec(ndim_);

    for (count_type i = 0; i < neval; ++i) {
      generate_point(point, grid_vec, i);
      const value_type func = point.weight * integrand(point);
      const value_type func2 = func * func;
      result_.accumulate(func, func2);
      for (size_type idim = 0; idim < ndim_; ++idim) {
        accumulator_(idim, grid_vec[idim]).accumulate(func2);
      }
    }

    return result_;
  }

  void reset() {
    std::vector<value_type> flat(ndiv_);
    for (size_type ig = 0; ig < ndiv_; ++ig) {
      flat[ig] = value_type(ig + 1) / value_type(ndiv_);
    }
    for (size_type idim = 0; idim < ndim_; ++idim) {
      for (size_type ig = 0; ig < ndiv_; ++ig) {
        grid_(idim, ig) = flat[ig];
      }
    }
    clear_data();
  }

  void adapt() {
    std::cout << "Adapting the grid on " << result_.count() << " collected samples.\n";
    for (auto idim = 0; idim < ndim_; ++idim) {
      /// initial values
      std::vector<value_type> d(ndiv_);
      std::vector<value_type> dval(ndiv_);
      value_type dsum = value_type(0);
      for (auto ig = 0; ig < ndiv_; ++ig) {
        /// Eq.(17) of https://arxiv.org/pdf/2009.05112
        if (accumulator_(idim, ig).count() == 0) {
          dval[ig] = value_type(0);
          continue;
        }
        dval[ig] =
            value_type(accumulator_(idim, ig).value()) / value_type(accumulator_(idim, ig).count());
        dsum += dval[ig];
      }

      if (dsum <= value_type(0)) {
        std::cout << "no data collected to adapt the grid" << std::endl;
        return;
      }

      /// smoothen out
      count_type nzero = 0;
      /// keep smoothing until zero-hit accumulators also have non-vanishing
      /// value
      for (size_type it = 0; it < nmax_smooth_; ++it) {
        dsum = value_type(0);
        nzero = 0;
        for (auto ig = 0; ig < ndiv_; ++ig) {
          if (ig == 0) {
            d[ig] = (7 * dval[ig] + dval[ig + 1]) / 8;
          } else if (ig == ndiv_ - 1) {
            d[ig] = (dval[ig - 1] + 7 * dval[ig]) / 8;
          } else {
            d[ig] = (dval[ig - 1] + 6 * dval[ig] + dval[ig + 1]) / 8;
          }
          /// accumulate & keep track of zero values
          dsum += d[ig];
          if ((accumulator_(idim, ig).count() == 0) && (d[ig] == value_type(0))) {
            nzero++;
          }
          // std::cout << idim << "[" << accumulator_(idim, ig).count() << "] " << d[ig] << "\n";
        }  // for ig
        /// copy back to dval for next smoothing iteration
        std::copy(d.begin(), d.end(), dval.begin());
        // std::cout << "smooth[" << it << "]:  # zero's = " << nzero << std::endl;
        if (nzero == 0) {
          break;
        }
      }  // for it

      /// normalize
      for (auto ig = 0; ig < ndiv_; ++ig) {
        d[ig] = dval[ig] / dsum;
      }

      /// dampen
      dsum = value_type(0);
      for (auto ig = 0; ig < ndiv_; ++ig) {
        if (d[ig] > value_type(0)) {
          d[ig] = std::pow((value_type(1) - d[ig]) / std::log(value_type(1) / d[ig]), alpha_);
        }
        dsum += d[ig];
      }

      /// refine the grid using `d`
      value_type davg = dsum / ndiv_;
      // std::cout << idim << ": " << dsum << "; <.> = " << davg << "\n";
      value_type dacc = 0.;
      size_type ig_new = 0;
      std::vector<value_type> g_new(ndiv_);
      for (auto ig = 0; ig < ndiv_; ++ig) {
        dacc += d[ig];
        // std::cout << "=- " << ig << ": " << dacc << " / " << davg << " -=\n";
        while (dacc >= davg) {
          dacc -= davg;
          value_type rat = dacc / d[ig];
          assert(rat >= 0.);
          assert(rat <= 1.);
          value_type x_low = ig > 0 ? grid_(idim, ig - 1) : 0.;
          value_type x_upp = grid_(idim, ig);
          g_new[ig_new] = x_low * rat + x_upp * (1. - rat);
          // std::cout << "  > " << ig_new << ": " << g_new[ig_new] << "\n";
          ig_new++;
        }
      }
      g_new[ndiv_ - 1] = 1.;

      ///--- check
      // std::cout << "CHK: ";
      for (auto ig = 0; ig < ndiv_; ++ig) {
        value_type x_new_low = ig > 0 ? g_new[ig - 1] : 0.;
        value_type x_new_upp = g_new[ig];
        value_type acc = 0.;
        for (auto jg = 0; jg < ndiv_; ++jg) {
          value_type x_old_low = jg > 0 ? grid_(idim, jg - 1) : 0.;
          value_type x_old_upp = grid_(idim, jg);
          if (x_old_upp < x_new_low) continue;
          if (x_old_low > x_new_upp) continue;
          value_type rat = (std::min(x_new_upp, x_old_upp) - std::max(x_new_low, x_old_low)) /
                           (x_old_upp - x_old_low);
          // std::cout << " + [" << x_old_low << "," << x_old_upp << "] " << rat << "\n";
          if (rat <= 0.) continue;
          acc += rat * d[jg];
        }
        // std::cout << "[" << x_new_low << "," << x_new_upp << "] " << acc << "\n";
        // std::cout << " " << acc;
      }
      // std::cout << std::endl;
      ///---

      for (size_type ig = 0; ig < ndiv_; ++ig) {
        grid_(idim, ig) = g_new[ig];
      }
    }  // for idim
    // print_grid();
    /// clear the accumulator to prepare for next iteration
    clear_data();
  }

  void clear_data() {
    for (size_type idim = 0; idim < ndim_; ++idim) {
      for (size_type ig = 0; ig < ndiv_; ++ig) {
        accumulator_(idim, ig).reset();
      }
    }
    result_.reset();
  }

  void print_grid() {
    for (auto i = 0; i < ndim_; ++i) {
      std::cout << "& " << i << "   " << 0. << " ";
      for (auto j = 0; j < ndiv_; ++j) {
        std::cout << grid_(i, j) << " ";
      }
      std::cout << "\n";
    }
  }

  void write_state_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    serialize_one<size_type>(out, ndim_);
    serialize_one<size_type>(out, ndiv_);
    grid_.serialize(out);
  }

  void read_state_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    deserialize_one<size_type>(in, ndim_);
    deserialize_one<size_type>(in, ndiv_);
    grid_ = ndarray::NDArray<value_type, size_type>({ndim_, ndiv_});
    grid_.deserialize(in);
    if (accumulator_.shape() != grid_.shape()) {
      accumulator_ = ndarray::NDArray<grid_acc_type, size_type>({ndim_, ndiv_});
    }
    // reset the result & accumulator
    clear_data();
  }

  void write_data_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    serialize_one<size_type>(out, ndim_);
    serialize_one<size_type>(out, ndiv_);
    serialize_one<kakuhen::util::HashValue_t>(out, hash().value());
    result_.serialize(out);
    accumulator_.serialize(out);
  }

  void read_data_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    //> check that we won't overwrite existing data
    if (result_.count() != 0) {
      throw std::runtime_error("result already has data");
    }
    for (size_type idim = 0; idim < ndim_; ++idim) {
      for (size_type ig = 0; ig < ndiv_; ++ig) {
        if (accumulator_(idim, ig).count() != 0) {
          throw std::runtime_error("accumulator already has data");
        }
      }
    }
    //> result and accumulator are empty; can just accumulate
    clear_data();  // for good measure
    accumulate_data_stream(in);
  }

  void accumulate_data_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    //> read & check for compatibility
    size_type ndim_chk;
    deserialize_one<size_type>(in, ndim_chk);
    if (ndim_chk != ndim_) {
      throw std::runtime_error("ndim mismatch");
    }
    size_type ndiv_chk;
    deserialize_one<size_type>(in, ndiv_chk);
    if (ndiv_chk != ndiv_) {
      throw std::runtime_error("ndiv mismatch");
    }
    if (grid_.shape() != std::vector<size_type>({ndim_, ndiv_})) {
      throw std::runtime_error("grid shape mismatch");
    }
    if (accumulator_.shape() != std::vector<size_type>({ndim_, ndiv_})) {
      throw std::runtime_error("accumulator shape mismatch");
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
    ndarray::NDArray<grid_acc_type, size_type> accumulator_in({ndim_, ndiv_});
    accumulator_in.deserialize(in);
    for (size_type idim = 0; idim < ndim_; ++idim) {
      for (size_type ig = 0; ig < ndiv_; ++ig) {
        accumulator_(idim, ig).accumulate(accumulator_in(idim, ig));
      }
    }
  }

 private:
  /// parameters that controls the grid refinement
  value_type alpha_ = value_type(1.2);
  size_type nmax_smooth_ = size_type(3);

  size_type ndiv_;  // number of divisions of the grid along each dimension
  ndarray::NDArray<value_type, size_type> grid_;
  int_acc_type result_;
  ndarray::NDArray<grid_acc_type, size_type> accumulator_;

  inline void generate_point(Point<num_traits>& point, std::vector<size_type>& grid_vec,
                             count_type sample_index = count_type(0)) {
    point.sample_index = sample_index;
    point.weight = value_type(1);
    for (size_type idim = 0; idim < ndim_; ++idim) {
      value_type rand = Base::ran();
      //> intervals rand in [ i/ndiv_ , (i+1)/ndiv_ ] mapped to i
      size_type ig = rand * ndiv_;  // 0 .. (ndiv_-1)
      assert(ig >= 0 && ig < ndiv_);
      assert(rand * ndiv_ >= value_type(ig) && rand * ndiv_ <= value_type(ig + 1));
      //> map rand back to [ 0, 1 ]
      rand = rand * ndiv_ - value_type(ig);
      assert(rand >= value_type(0) && rand <= value_type(1));
      value_type x_low = ig > 0 ? grid_(idim, ig - 1) : value_type(0);
      value_type x_upp = grid_(idim, ig);
      point.x[idim] = x_low + rand * (x_upp - x_low);
      // point.x[idim] = x_low * (T(1) - rand) + x_upp * rand;
      grid_vec[idim] = ig;
      point.weight *= ndiv_ * (x_upp - x_low);
    }
  }

};  // class Vegas

}  // namespace kakuhen::integrator
