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
  //> some shorthands to save typing
  using S = typename Base::size_type;
  using T = typename Base::value_type;
  using U = typename Base::count_type;
  using grid_acc_type = GridAccumulator<T, U>;
  //  member variables
  using Base::ndim_;
  using Base::opts_;

  explicit Vegas(S ndim, S ndiv = 512)
      : Base(ndim), ndiv_{ndiv}, grid_({ndim, ndiv}), accumulator_({ndim, ndiv}) {
    assert(ndim > 0 && ndiv > 2);
    reset();
  };

  explicit Vegas(const std::filesystem::path& filepath) : Base(0) {
    Base::load(filepath);
  }

  inline void set_alpha(T alpha) noexcept {
    assert(alpha >= T(0));
    alpha_ = alpha;
  }
  inline T alpha() const noexcept {
    return alpha_;
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
  int_acc_type integrate_impl(I&& integrand, U neval) {
    result_.reset();

    Point<num_traits> point{ndim_, opts_.user_data.value_or(nullptr)};

    std::vector<S> grid_vec(ndim_);

    for (U i = 0; i < neval; ++i) {
      generate_point(point, grid_vec, i);
      const T func = point.weight * integrand(point);
      const T func2 = func * func;
      const T abs_func = std::abs(func);
      result_.accumulate(func, func2);
      for (S idim = 0; idim < ndim_; ++idim) {
        ///broken w/o damping:  accumulator_(idim, grid_vec[idim]).accumulate(func2);
        ///works: accumulator_(idim, grid_vec[idim]).accumulate(func2/point.weight);
        accumulator_(idim, grid_vec[idim]).accumulate(abs_func);
      }
    }

    return result_;
  }

  void reset() {
    grid_.fill(T(0));
    std::vector<T> flat(ndiv_);
    for (S ig = 0; ig < ndiv_; ++ig) {
      flat[ig] = T(ig + 1) / T(ndiv_);
    }
    for (S idim = 0; idim < ndim_; ++idim) {
      for (S ig = 0; ig < ndiv_; ++ig) {
        grid_(idim, ig) = flat[ig];
      }
    }
    clear_data();
  }

  void adapt() {
    // using kakuhen::ndarray::_;
    using kakuhen::ndarray::NDArray;
    using kakuhen::ndarray::NDView;

    std::cout << "Adapting the grid on " << result_.count() << " collected samples.\n";

    //> pre-allocate data structures
    NDArray<T, S> dval({ndiv_});
    NDArray<T, S> d({ndiv_});
    NDArray<T, S> grid_new({ndiv_});
    T dsum, davg, dacc;

    for (S idim = 0; idim < ndim_; ++idim) {
      /// initialize
      dval.fill(T(0));
      dsum = T(0);
      for (S ig = 0; ig < ndiv_; ++ig) {
        if (accumulator_(idim, ig).count() == 0) {
          dval(ig) = T(0);
          continue;
        }
        dval(ig) = accumulator_(idim, ig).value() / T(accumulator_(idim, ig).count());
        dsum += dval(ig);
      }

      if (dsum <= T(0)) {
        std::cout << "no data collected to adapt the grid" << std::endl;
        return;
      }

      /// smoothen out
      d.fill(T(0));
      dsum = T(0);
      for (auto ig = 0; ig < ndiv_; ++ig) {
        if (ig == 0) {
          d(ig) = (7 * dval(ig) + dval(ig + 1)) / T(8);
        } else if (ig == ndiv_ - 1) {
          d(ig) = (dval(ig - 1) + 7 * dval(ig)) / T(8);
        } else {
          d(ig) = (dval(ig - 1) + 6 * dval(ig) + dval(ig + 1)) / T(8);
        }
        dsum += d(ig);
      }  // for ig

      /// normalize
      for (auto ig = 0; ig < ndiv_; ++ig) {
        d(ig) = d(ig) / dsum;
      }

      /// dampen
      dsum = T(0);
      for (auto ig = 0; ig < ndiv_; ++ig) {
        if (d(ig) > T(0)) {
          d(ig) = std::pow(-(T(1) - d(ig)) / std::log(d(ig)), alpha_);
        }
        dsum += d(ig);
      }

      /// refine the grid using `d`
      davg = dsum / T(ndiv_);
      // std::cout << idim << ": " << dsum << "; <.> = " << davg << "\n";
      dacc = T(0);
      S ig_new = S(0);
      for (S ig = 0; ig < ndiv_; ++ig) {
        dacc += d(ig);
        // std::cout << "=- " << ig << ": " << dacc << " / " << davg << " -=\n";
        while (dacc >= davg) {
          dacc -= davg;
          const T rat = dacc / d(ig);
          assert(rat >= 0. && rat <= 1.);
          const T x_low = ig > 0 ? grid_(idim, ig - 1) : 0.;
          const T x_upp = grid_(idim, ig);
          grid_new(ig_new) = x_low * rat + x_upp * (1. - rat);
          // std::cout << "  > " << ig_new << ": " << grid_new(ig_new) << "\n";
          ig_new++;
        }
      }
      grid_new(ndiv_ - 1) = T(1);

      ///--- check
      // std::cout << "CHK: ";
      for (S ig = 0; ig < ndiv_; ++ig) {
        T x_new_low = ig > 0 ? grid_new[ig - 1] : 0.;
        T x_new_upp = grid_new[ig];
        T acc = T(0);
        for (S jg = 0; jg < ndiv_; ++jg) {
          T x_old_low = jg > 0 ? grid_(idim, jg - 1) : 0.;
          T x_old_upp = grid_(idim, jg);
          if (x_old_upp < x_new_low) continue;
          if (x_old_low > x_new_upp) continue;
          T rat = (std::min(x_new_upp, x_old_upp) - std::max(x_new_low, x_old_low)) /
                  (x_old_upp - x_old_low);
          // std::cout << " + [" << x_old_low << "," << x_old_upp << "] " << rat << "\n";
          if (rat <= 0.) continue;
          acc += rat * d(jg);
        }
        // std::cout << "[" << x_new_low << "," << x_new_upp << "] " << acc << "\n";
        // std::cout << " " << acc;
      }
      // std::cout << std::endl;
      ///---

      for (S ig = 0; ig < ndiv_; ++ig) {
        grid_(idim, ig) = grid_new[ig];
      }
    }  // for idim
    // print_grid();
    /// clear the accumulator to prepare for next iteration
    clear_data();
  }

  void clear_data() {
    std::for_each(accumulator_.begin(), accumulator_.end(), [](auto& acc) { acc.reset(); });
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
    if (accumulator_.shape() != grid_.shape()) {
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
    accumulator_.serialize(out);
  }

  void read_data_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    //> check that we won't overwrite existing data
    if (result_.count() != 0) {
      throw std::runtime_error("result already has data");
    }
    for (S idim = 0; idim < ndim_; ++idim) {
      for (S ig = 0; ig < ndiv_; ++ig) {
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
    if (grid_.shape() != std::vector<S>({ndim_, ndiv_})) {
      throw std::runtime_error("grid shape mismatch");
    }
    if (accumulator_.shape() != std::vector<S>({ndim_, ndiv_})) {
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
    ndarray::NDArray<grid_acc_type, S> accumulator_in({ndim_, ndiv_});
    accumulator_in.deserialize(in);
    for (S idim = 0; idim < ndim_; ++idim) {
      for (S ig = 0; ig < ndiv_; ++ig) {
        accumulator_(idim, ig).accumulate(accumulator_in(idim, ig));
      }
    }
  }

 private:
  /// parameters that controls the grid refinement
  T alpha_{0.75};

  S ndiv_;  // number of divisions of the grid along each dimension
  ndarray::NDArray<T, S> grid_;
  int_acc_type result_;
  ndarray::NDArray<grid_acc_type, S> accumulator_;

  inline void generate_point(Point<num_traits>& point, std::vector<S>& grid_vec,
                             U sample_index = U(0)) {
    point.sample_index = sample_index;
    point.weight = T(1);
    for (S idim = 0; idim < ndim_; ++idim) {
      T rand = Base::ran();
      //> intervals rand in [ i/ndiv_ , (i+1)/ndiv_ ] mapped to i
      const S ig = S(rand * ndiv_);  // 0 .. (ndiv_-1)
      assert(ig >= 0 && ig < ndiv_);
      assert(rand * ndiv_ >= T(ig) && rand * ndiv_ <= T(ig + 1));
      //> map rand back to [ 0, 1 ]
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
