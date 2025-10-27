/// Instead of a *function*, the integrator also accepts a *functor*
/// this is a class/struct that has `operator()` implemented, i.e.
/// it's objects can be called like functions.
///
/// The advantage is that this class can encapsulate datastructures
/// & methods that aid .e.g. in switching between modes and
/// populating histograms.
///
/// Here's a minimal example to illustate such use-cases

#include "kakuhen/kakuhen.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace kakuhen::integrator;

class MyFunctor {
 public:
  MyFunctor() : stage_{0}, count_{0} {
    /// allocate the histgram datastructure
    /// in this example we lay out the data of 3 histograms like this:
    ///   (a) 0-9:   10 bins in x[0]
    ///   (b) 10-19: 10 bins in x[1]
    ///   (c) 20-39: 20 bins in y == x[0]+x[1]
    ///   (d) 40:    1 bin for the full integral
    /// more sophisticated implementations possible with dynamical allocation
    /// through e.g. `register_histogram` routines etc.
    /// we keep it simple here for the general idea
    histogram_data_ = std::vector<HistogramBin_t>{41};
    reset_histogram();
  }

  /// the funciton in 2D to be integrated
  double operator()(const Point<>& point) {
    assert(point.ndim == 2);
    const auto& x = point.x;  // shorthand

    const std::vector<double> r1 = {0.15, 0.2};
    const std::vector<double> r2 = {0.65, 0.5};

    double dr1 = 0.;
    double dr2 = 0.;
    for (auto i = 0; i < 2; ++i) {
      dr1 += (x[i] - r1[i]) * (x[i] - r1[i]);
      dr2 += (x[i] - r2[i]) * (x[i] - r2[i]);
    }

    double fval = 1e3 * std::exp(-50 * dr1) + 7e2 * std::exp(-20 * dr2);

    /// if we're in production mode, bin the event to the histograms
    /// note: accumulate the function times the weight of the event
    if (stage_ > 0) {
      bin_histogram(x, fval * point.weight);
    }
    return fval;
  }

  /// get/set the stage of the calculation
  inline int stage() const noexcept {
    return stage_;
  }
  inline void set_stage(int stage) noexcept {
    stage_ = stage;
  }

  /// reset all accumulated histogram information
  void reset_histogram() {
    count_ = 0;
    std::for_each(histogram_data_.begin(), histogram_data_.end(), [](auto& acc) {
      acc.value = 0.;
      acc.valuesq = 0.;
    });
  }

  /// "bin" the event `x` to the histograms
  void bin_histogram(const std::vector<double>& x, const double& val) {
    const double valsq = val * val;
    /// increment counter
    count_++;
    /// populate histogram (a)
    /// x[0] from 0,1 in 10 bins, offset = 0
    auto ibin = 10 * x[0];
    histogram_data_.at(ibin).value += val;
    histogram_data_.at(ibin).valuesq += valsq;
    /// populate histogram (b)
    /// x[1] from 0,1 in 10 bins, offset = 10
    ibin = 10 + 10 * x[1];
    histogram_data_.at(ibin).value += val;
    histogram_data_.at(ibin).valuesq += valsq;
    /// populate histogram (c)
    /// y == (x[0]+x[1]) from 0,2 in 20 bins, offset = 20
    const double y = x[0] + x[1];
    ibin = 20 + 20 * (y / 2);
    histogram_data_.at(ibin).value += val;
    histogram_data_.at(ibin).valuesq += valsq;
    /// populate "histogram" (d)
    /// total integral
    ibin = 40;
    histogram_data_.at(ibin).value += val;
    histogram_data_.at(ibin).valuesq += valsq;
  }

  void print_histogram() {
    /// loop over all histogram data bins in one go
    double sum_val = 0.;
    double sum_err = 0.;
    for (auto ibin = 0; ibin < histogram_data_.size(); ++ibin) {
      /// histogram headers:
      if (ibin == 0) std::cout << "\n\n# histogram (a) --- x[0]\n";
      if (ibin == 10) std::cout << "\n\n# histogram (b) --- x[1]\n";
      if (ibin == 20) std::cout << "\n\n# histogram (c) --- y == x[0]+x[1]\n";
      if (ibin == 40) std::cout << "\n\n# histogram (d) --- total integral\n";
      if ((ibin == 0) || (ibin == 10) || (ibin == 20) || (ibin == 40)) {
        sum_val = 0.;
        sum_err = 0.;
      }
      /// compute x-values of the bin (lower & upper edges) & offsets
      int jbin;
      double xlow, xupp;
      if (ibin < 10) {
        jbin = ibin;
        xlow = double(jbin) / 10.;
        xupp = double(jbin + 1) / 10.;
      } else if (ibin < 20) {
        jbin = ibin - 10;
        xlow = double(jbin) / 10.;
        xupp = double(jbin + 1) / 10.;
      } else if (ibin < 40) {
        jbin = ibin - 20;
        xlow = double(jbin) / 10.;
        xupp = double(jbin + 1) / 10.;
      } else if (ibin == 40) {
        jbin = 0;
        xlow = 0.;
        xupp = 1.;
      } else {
        throw "invalid index";
      }
      /// shorthands that the compiler will optimize out
      const double& sumf = histogram_data_.at(ibin).value;
      const double& sumf2 = histogram_data_.at(ibin).valuesq;
      /// result and stddev for the histogram bin
      /// note: we do *not* divide by the bin width!
      /// to get "df/dO", you will want to divide by (xupp-xlow)
      const double res = sumf / double(count_);
      const double err =
          count_ > 0 ? ((sumf2 / double(count_) - res * res) / double(count_ - 1)) : 0.;
      /// output the row of the histogram bin:
      /// [1] bin idx, [2,3] bin range, [4] value, [5] error
      std::cout << jbin << "   " << xlow << " " << xupp << "   ";
      std::cout << res << " " << std::sqrt(err) << "\n";
      /// the accumulator & footer write out
      sum_val += res;
      sum_err += err;
      if (ibin == 9)
        std::cout << "#Σ " << sum_val << " +/- " << std::sqrt(sum_err) << " [" << count_ << "]\n";
      if (ibin == 19)
        std::cout << "#Σ " << sum_val << " +/- " << std::sqrt(sum_err) << " [" << count_ << "]\n";
      if (ibin == 39)
        std::cout << "#Σ " << sum_val << " +/- " << std::sqrt(sum_err) << " [" << count_ << "]\n";
      if (ibin == 40)
        std::cout << "#Σ " << sum_val << " +/- " << std::sqrt(sum_err) << " [" << count_ << "]\n";
    }
  }

 private:
  /// a flag to control the stage of the calculation
  /// in this example:
  /// stage_ = 0:  warmup phase (switch off histograms)
  /// stage_ = 1:  production phase
  int stage_;
  /// POD for a single histogram bin; for accumulators including
  /// compensated summation, check out `accumulator.h`
  uint32_t count_ = 0;
  struct HistogramBin_t {
    double value = 0.;
    double valuesq = 0.;
  };
  /// use a simple vector to store the histogram data
  /// mappings to (multi-dimensional) histograms can be
  /// done with a simple wrapper class with strides etc.
  std::vector<HistogramBin_t> histogram_data_;
};

int main() {
  /// load the namespace for convenience
  using namespace kakuhen::integrator;

  /// initialize a `MyFunctor` object
  MyFunctor integrand{};

  auto integrator = Basin(2);  // 2 dimensions
  integrand.set_stage(0);      // warmup: switch off histogram filling
  integrator.integrate(integrand, {.neval = 50000, .niter = 7, .adapt = true});
  integrator.set_options({.adapt = false});  // freeze the grid -> production phase
  integrand.set_stage(1);                    // production with histogram filling
  integrand.reset_histogram();               // for good measure
  auto result = integrator.integrate(integrand, {.neval = 1000000, .niter = 3, .verbosity = 1});
  std::cout << "integral = " << result.value() << " +/- " << result.error() << "\n";
  integrand.print_histogram();

  return 0;
}
