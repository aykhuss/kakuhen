// basin - Blockwise Adaptive Sampling with Interdimensional Nesting
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
  // using typename Base::result_type;
  using grid_acc_type = GridAccumulator<value_type, count_type>;
  //  member variables
  using Base::ndim_;
  using Base::opts_;

  explicit Basin(size_type ndim, size_type ndiv1 = 16, size_type ndiv2 = 32)
      : Base(ndim),
        ndiv1_{ndiv1},
        ndiv2_{ndiv2},
        ndiv0_{ndiv1 * ndiv2},
        grid_({ndim, ndim, ndiv1, ndiv2}),
        accumulator_({ndim, ndim, ndiv1, ndiv2}),
        order_({ndim, 2}) {
    assert(ndim > 0 && ndiv1 > 2 && ndiv2 > 2);
    grid0_ = grid_.view().diagonal(0, 1).reshape({ndim_, ndiv0_});
    accumulator0_ = accumulator_.view().diagonal(0, 1).reshape({ndim_, ndiv0_});
    reset();
  };

  explicit Basin(const std::filesystem::path& filepath) : Base(0) {
    Base::load(filepath);
  }

  inline void set_alpha(value_type alpha) noexcept {
    assert(alpha >= value_type(0));
    alpha_ = alpha;
  }
  inline value_type alpha() const noexcept {
    return alpha_;
  }

  inline void set_nmax_smooth(size_type nmax_smooth) noexcept {
    nmax_smooth_ = nmax_smooth;
  }
  inline size_type nmax_smooth() const noexcept {
    return nmax_smooth_;
  }

  inline kakuhen::util::Hash hash() const {
    return kakuhen::util::Hash().add(ndim_).add(ndiv1_).add(ndiv2_).add(grid_.data(), grid_.size());
  }

  inline std::string prefix(bool with_hash = false) const noexcept {
    std::string pref = "basin_" + std::to_string(ndim_) + "d";
    if (with_hash) pref += "_" + hash().encode_hex();
    return pref;
  }

  template <typename I>
  int_acc_type integrate_impl(I&& integrand, count_type neval) {
    result_.reset();

    Point<num_traits> point{ndim_, opts_.user_data.value_or(nullptr)};

    std::vector<size_type> grid_vec(ndim_);  // vetor in `ndiv0_` space

    for (count_type i = 0; i < neval; ++i) {
      generate_point(point, grid_vec, i);
      const value_type func = point.weight * integrand(point);
      const value_type func2 = func * func;
      result_.accumulate(func, func2);
      for (size_type idim = 0; idim < ndim_; ++idim) {
        accumulator0_(idim, grid_vec[idim]).accumulate(func2);
      }
    }

    return result_;
  }

  void reset() {
    grid_.fill(value_type(0));
    /// first the non-diagonal entries
    std::vector<value_type> flat2(ndiv2_);
    for (size_type ig2 = 0; ig2 < ndiv2_; ++ig2) {
      flat2[ig2] = value_type(ig2 + 1) / value_type(ndiv2_);
    }
    for (size_type idim1 = 0; idim1 < ndim_; ++idim1) {
      for (size_type idim2 = 0; idim2 < ndim_; ++idim2) {
        if (idim1 == idim2) continue;
        for (size_type ig1 = 0; ig1 < ndiv1_; ++ig1) {
          for (size_type ig2 = 0; ig2 < ndiv2_; ++ig2) {
            grid_(idim1, idim2, ig1, ig2) = flat2[ig2];
          }
        }
      }
    }
    /// the diagonal entries
    std::vector<value_type> flat0(ndiv0_);
    for (size_type ig0 = 0; ig0 < ndiv0_; ++ig0) {
      flat0[ig0] = value_type(ig0 + 1) / value_type(ndiv0_);
    }
    for (size_type idim = 0; idim < ndim_; ++idim) {
      for (size_type ig0 = 0; ig0 < ndiv0_; ++ig0) {
        grid0_(idim, ig0) = flat0[ig0];
      }
    }
    /// initialize the sampling order
    order_.fill(0);
    for (size_type idim = 0; idim < ndim_; ++idim) {
      order_(idim, 0) = idim;
      order_(idim, 1) = idim;
    }
    /// also cear the accumulators
    clear_data();
  }

  void adapt() {
    std::cout << "Adapting the grid on " << result_.count() << " collected samples.\n";

    for (auto idim1 = 0; idim1 < ndim_; ++idim1) {
      /// initial values
      std::vector<value_type> d0(ndiv0_);
      std::vector<value_type> d0val(ndiv0_);
      value_type d0sum = value_type(0);
      for (auto ig0 = 0; ig0 < ndiv0_; ++ig0) {
        if (accumulator0_(idim1, ig0).count() == 0) {
          d0val[ig0] = value_type(0);
          continue;
        }
        d0val[ig0] = accumulator0_(idim1, ig0).value() / value_type(accumulator0_(idim1, ig0).count());
        d0sum += d0val[ig0];
      }

      if (d0sum <= value_type(0)) {
        std::cout << "no data collected to adapt the grid" << std::endl;
        return;
      }

      /// smoothen out
      size_type nzero = 0;
      /// keep smoothing until zero-hit accumulators also have non-vanishing value
      for (size_type it = 0; it < nmax_smooth_; ++it) {
        d0sum = value_type(0);
        nzero = 0;
        for (auto ig0 = 0; ig0 < ndiv0_; ++ig0) {
          if (ig0 == 0) {
            d0[ig0] = (7 * d0val[ig0] + d0val[ig0 + 1]) / value_type(8);
          } else if (ig0 == ndiv0_ - 1) {
            d0[ig0] = (d0val[ig0 - 1] + 7 * d0val[ig0]) / value_type(8);
          } else {
            d0[ig0] = (d0val[ig0 - 1] + 6 * d0val[ig0] + d0val[ig0 + 1]) / value_type(8);
          }
          /// accumulate & keep track of zero values
          d0sum += d0[ig0];
          if ((accumulator0_(idim1, ig0).count() == 0) && (d0[ig0] == value_type(0))) {
            nzero++;
          }
          // std::cout << idim1 << "[" << accumulator0_(idim1, ig0).count() << "] " << d0[ig0] << "\n";
        }  // for ig0
        /// copy back to d0val for next smoothing iteration
        std::copy(d0.begin(), d0.end(), d0val.begin());
        // std::cout << "smooth[" << it << "]:  # zero's = " << nzero << std::endl;
        if (nzero == 0) {
          break;
        }
      }  // for it

      /// normalize
      for (auto ig0 = 0; ig0 < ndiv0_; ++ig0) {
        d0[ig0] = d0val[ig0] / d0sum;
        // std::cout << ig0 << ": check = " << ndiv0_ * d0[ig0] << "\n";  // should converge towards ->
        // 1
      }

      /// dampen
      d0sum = value_type(0);
      for (auto ig0 = 0; ig0 < ndiv0_; ++ig0) {
        if (d0[ig0] > value_type(0)) {
          d0[ig0] = std::pow(-(value_type(1) - d0[ig0]) / std::log(d0[ig0]), alpha_);
        }
        d0sum += d0[ig0];
      }

      /// refine the grid using `d`
      value_type davg = d0sum / value_type(ndiv0_);
      // std::cout << idim1 << ": " << d0sum << "; <.> = " << davg << "\n";
      value_type dacc = 0.;
      size_type ig0_new = 0;
      std::vector<value_type> g_new(ndiv0_, 0);
      for (auto ig0 = 0; ig0 < ndiv0_; ++ig0) {
        dacc += d0[ig0];
        // std::cout << "=- " << ig0 << ": " << dacc << " / " << davg << " -=\n";
        while (dacc >= davg) {
          dacc -= davg;
          value_type rat = dacc / d0[ig0];
          assert(rat >= 0.);
          assert(rat <= 1.);
          value_type x_low = ig0 > 0 ? grid0_(idim1, ig0 - 1) : 0.;
          value_type x_upp = grid0_(idim1, ig0);
          g_new[ig0_new] = x_low * rat + x_upp * (1. - rat);
          // std::cout << "  > " << ig0_new << ": " << g_new[ig0_new] << "\n";
          ig0_new++;
        }
      }
      g_new[ndiv0_ - 1] = 1.;

      for (size_type ig0 = 0; ig0 < ndiv0_; ++ig0) {
        grid0_(idim1, ig0) = g_new[ig0];
      }

      ////// idim2 optimization


    }  // for idim1

    ////// determine the sampling order


    /// clear the accumulator to prepare for next iteration
    clear_data();
  }

  void clear_data() {
    for (size_type idim1 = 0; idim1 < ndim_; ++idim1) {
      for (size_type idim2 = 0; idim2 < ndim_; ++idim2) {
        for (size_type ig1 = 0; ig1 < ndiv1_; ++ig1) {
          for (size_type ig2 = 0; ig2 < ndiv2_; ++ig2) {
            accumulator_(idim1, idim2, ig1, ig2).reset();
          }
        }
      }
    }
    result_.reset();
  }

  void print_grid() {}

  void write_state_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    serialize_one<size_type>(out, ndim_);
    serialize_one<size_type>(out, ndiv1_);
    serialize_one<size_type>(out, ndiv2_);
    grid_.serialize(out);
    order_.serialize(out);
  }

  void read_state_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    deserialize_one<size_type>(in, ndim_);
    deserialize_one<size_type>(in, ndiv1_);
    deserialize_one<size_type>(in, ndiv2_);
    ndiv0_ = ndiv1_ * ndiv2_;
    grid_ = ndarray::NDArray<value_type, size_type>({ndim_, ndim_, ndiv1_, ndiv2_});
    grid_.deserialize(in);
    grid0_ = grid_.view().diagonal(0, 1).reshape({ndim_, ndiv0_});
    if (accumulator_.shape() != grid_.shape() || accumulator0_.shape() != grid0_.shape()) {
      accumulator_ = ndarray::NDArray<grid_acc_type, size_type>({ndim_, ndim_, ndiv1_, ndiv2_});
      accumulator0_ = accumulator_.view().diagonal(0, 1).reshape({ndim_, ndiv0_});
    }
    order_ = ndarray::NDArray<size_type, size_type>({ndim_, 2});
    order_.deserialize(in);
    /// reset the result & accumulator
    clear_data();
  }

  void write_data_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    serialize_one<size_type>(out, ndim_);
    serialize_one<size_type>(out, ndiv1_);
    serialize_one<size_type>(out, ndiv2_);
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
    for (size_type idim1 = 0; idim1 < ndim_; ++idim1) {
      for (size_type idim2 = 0; idim2 < ndim_; ++idim2) {
        for (size_type ig1 = 0; ig1 < ndiv1_; ++ig1) {
          for (size_type ig2 = 0; ig2 < ndiv2_; ++ig2) {
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

  void accumulate_data_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    //> read & check for compatibility
    size_type ndim_chk;
    deserialize_one<size_type>(in, ndim_chk);
    if (ndim_chk != ndim_) {
      throw std::runtime_error("ndim mismatch");
    }
    size_type ndiv1_chk;
    deserialize_one<size_type>(in, ndiv1_chk);
    if (ndiv1_chk != ndiv1_) {
      throw std::runtime_error("ndiv1 mismatch");
    }
    size_type ndiv2_chk;
    deserialize_one<size_type>(in, ndiv2_chk);
    if (ndiv2_chk != ndiv2_) {
      throw std::runtime_error("ndiv2 mismatch");
    }
    if (grid_.shape() != std::vector<size_type>({ndim_, ndim_, ndiv1_, ndiv2_})) {
      throw std::runtime_error("grid shape mismatch");
    }
    if (grid0_.shape() != std::vector<size_type>({ndim_, ndiv0_})) {
      throw std::runtime_error("grid0 shape mismatch");
    }
    if (accumulator_.shape() != std::vector<size_type>({ndim_, ndim_, ndiv1_, ndiv2_})) {
      throw std::runtime_error("accumulator shape mismatch");
    }
    if (accumulator0_.shape() != std::vector<size_type>({ndim_, ndiv0_})) {
      throw std::runtime_error("accumulator0 shape mismatch");
    }
    if (order_.shape() != std::vector<size_type>({ndim_, 2})) {
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
    ndarray::NDArray<grid_acc_type, size_type> accumulator_in({ndim_, ndim_, ndiv1_, ndiv2_});
    accumulator_in.deserialize(in);
    for (size_type idim1 = 0; idim1 < ndim_; ++idim1) {
      for (size_type idim2 = 0; idim2 < ndim_; ++idim2) {
        for (size_type ig1 = 0; ig1 < ndiv1_; ++ig1) {
          for (size_type ig2 = 0; ig2 < ndiv2_; ++ig2) {
            accumulator_(idim1, idim2, ig1, ig2).accumulate(accumulator_in(idim1, idim2, ig1, ig2));
          }
        }
      }
    }
  }

 private:
  /// parameters that controls the grid refinement
  value_type alpha_{0.75};
  size_type nmax_smooth_{3};

  /// division for conditional PDF:  P(x2|x1)
  size_type ndiv1_;  // number of divisions of the grid along dim 1
  size_type ndiv2_;  // number of divisions of the grid along dim 2
  size_type ndiv0_;  // number of divisions of the grid along diagonal
  ndarray::NDArray<value_type, size_type> grid_;
  ndarray::NDView<value_type, size_type> grid0_;
  int_acc_type result_;
  ndarray::NDArray<grid_acc_type, size_type> accumulator_;
  ndarray::NDView<grid_acc_type, size_type> accumulator0_;
  /// define the sampling order
  ndarray::NDArray<value_type, size_type> order_;

  inline void generate_point(Point<num_traits>& point, std::vector<size_type>& grid_vec,
                             count_type sample_index = count_type(0)) {
    point.sample_index = sample_index;
    point.weight = value_type(1);
    for (size_type iord = 0; iord < ndim_; ++iord) {
      value_type rand = Base::ran();
      if (order_(iord, 0) == order_(iord, 1)) {
        /// (a) diagnoal map
        const size_type idim0 = order_(iord, 0);
        //> intervals rand in [ i/ndiv0_ , (i+1)/ndiv0_ ] mapped to i
        const size_type ig0 = rand * ndiv0_;  // 0 .. (ndiv0_-1)
        assert(ig0 >= 0 && ig0 < ndiv0_);
        assert(rand * ndiv0_ >= value_type(ig0) && rand * ndiv0_ <= value_type(ig0 + 1));
        //> map rand back to [ 0, 1 ]
        rand = rand * ndiv0_ - value_type(ig0);
        assert(rand >= value_type(0) && rand <= value_type(1));
        const value_type x_low = ig0 > 0 ? grid0_(idim0, ig0 - 1) : value_type(0);
        const value_type x_upp = grid0_(idim0, ig0);
        point.x[idim0] = x_low + rand * (x_upp - x_low);
        // point.x[idim0] = x_low * (T(1) - rand) + x_upp * rand;
        point.weight *= ndiv0_ * (x_upp - x_low);
        grid_vec[idim0] = ig0;

      } else {
        /// (b) conditional map
        const size_type idim1 = order_(iord, 0);
        const size_type idim2 = order_(iord, 1);
        /// check that the 1st dimension is set properly
        assert(point.x[idim1] >= value_type(0) && point.x[idim1] <= value_type(1));
        assert(grid_vec[idim1] >= 0 && grid_vec[idim1] < ndiv0_);
        /// this is correct because `grid_vec` always stores `ig0`
        const size_type ig1 = static_cast<size_type>(grid_vec[idim1] / ndiv2_);
        assert(ig1 >= 0 && ig1 < ndiv1_);
        //> intervals rand in [ i/ndiv2_ , (i+1)/ndiv2_ ] mapped to i
        const size_type ig2 = rand * ndiv2_;  // 0 .. (ndiv2_-1)
        assert(ig2 >= 0 && ig2 < ndiv2_);
        assert(rand * ndiv2_ >= value_type(ig2) && rand * ndiv2_ <= value_type(ig2 + 1));
        //> map rand back to [ 0, 1 ]
        rand = rand * ndiv2_ - value_type(ig2);
        assert(rand >= value_type(0) && rand <= value_type(1));
        const value_type x_low = ig2 > 0 ? grid_(idim1, idim2, ig1, ig2 - 1) : value_type(0);
        const value_type x_upp = grid_(idim1, idim2, ig1, ig2);
        const value_type x = x_low + rand * (x_upp - x_low);
        point.x[idim2] = x;
        // point.x[idim2] = x_low * (T(1) - rand) + x_upp * rand;
        point.weight *= ndiv2_ * (x_upp - x_low);
        /// need to get index ig0 for idim2
        size_type ig0 = 0;
        size_type ig0_hi = ndiv0_;
        while (ig0 < ig0_hi) {
          const size_type mid =
              ig0 + ((ig0_hi - ig0) >> 1);  // same as `(ig0 + ig0_hi)/2` but safer against overflow
          if (x < grid0_(idim2, mid))
            ig0_hi = mid;
          else
            ig0 = mid + 1;
        }
        assert(ig0 >= 0 && ig0 < ndiv0_);
        assert(x >= grid0_(idim2, ig0 - 1) && x <= grid0_(idim2, ig0));
        grid_vec[idim2] = ig0;
      }
    }
  }

};  // class Basin

}  // namespace kakuhen::integrator
