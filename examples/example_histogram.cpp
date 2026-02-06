/// Instead of a *function*, the integrator also accepts a *functor*
/// this is a class/struct that has `operator()` implemented, i.e.
/// it's objects can be called like functions.
///
/// The advantage is that this class can encapsulate datastructures
/// & methods that aid .e.g. in switching between modes and
/// populating histograms.
///
/// Here's a minimal example to illustate such use-cases

#include "kakuhen/histogram/axis.h"
#include "kakuhen/histogram/histogram_registry.h"
#include "kakuhen/histogram/histogram_writer.h"
#include "kakuhen/kakuhen.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace kakuhen;
using namespace integrator;
using namespace histogram;

/// default numeric traits
using NT = util::num_traits_t<>;
using S = typename NT::size_type;
using T = typename NT::value_type;
using U = typename NT::count_type;

class MyFunctor {
 public:
  using UniformAxis_t = UniformAxis<T, S>;
  using VariableAxis_t = VariableAxis<T, S>;
  using registry_type = HistogramRegistry<NT>;
  using buffer_type = HistogramBuffer<NT>;
  using id_type = registry_type::Id;

  MyFunctor() : stage_{0}, registry_{}, buffer_{} {
    /// define histograms and set up buffer
    hist_x = registry_.book("x", 1, UniformAxis_t(10, 0., 1.));
    hist_y = registry_.book("y", 1, UniformAxis_t(10, 0., 1.));
    hist_xy = registry_.book("x+y", 1, UniformAxis_t(20, 0., 2.));
    buffer_ = registry_.create_buffer();
  }

  /// the funciton in 2D to be integrated
  T operator()(const Point<NT>& point) {
    assert(point.ndim == 3);

    const auto& x = point.x;  // shorthand

    const std::vector<T> r1 = {0.15, 0.2};
    const std::vector<T> r2 = {0.65, 0.5};

    T dr1 = 0.;
    T dr2 = 0.;
    for (S i = 0; i < 2; ++i) {
      dr1 += (x[i] - r1[i]) * (x[i] - r1[i]);
      dr2 += (x[i] - r2[i]) * (x[i] - r2[i]);
    }

    T fval = 1e3 * std::exp(-50 * dr1) + 7e2 * std::exp(-20 * dr2);

    /// diagonal structure
    const T off_diag = std::fabs(x[1] - x[2]);
    fval *= std::exp(-20. * off_diag * off_diag);

    /// if we're in production mode, bin the event to the histograms
    /// note: accumulate the function times the weight of the event
    if (stage_ > 0) {
      const T hist_wgt = fval * point.weight;
      registry_.fill(buffer_, hist_x, hist_wgt, x[0]);
      registry_.fill(buffer_, hist_y, hist_wgt, x[1]);
      registry_.fill(buffer_, hist_xy, hist_wgt, x[0] + x[1]);
      registry_.flush(buffer_);
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

  void print_histogram() {
    NNLOJETWriter<NT> writer(std::cout);
    registry_.write(writer);
  }

 private:
  /// a flag to control the stage of the calculation
  /// stage_ = 0:  warmup phase (switch off histograms)
  /// stage_ = 1:  production phase
  int stage_;
  registry_type registry_;
  buffer_type buffer_;
  id_type hist_x, hist_y, hist_xy;
};

int main() {
  /// initialize a `MyFunctor` object
  MyFunctor integrand{};

  auto integrator = Vegas(3);  // 3 dimensions
  integrand.set_stage(0);      // warmup: switch off histogram filling
  integrator.integrate(integrand, {.neval = 50000, .niter = 7, .adapt = true});
  integrator.set_options({.adapt = false});  // freeze the grid -> production phase
  integrand.set_stage(1);                    // production with histogram filling
  integrator.save();
  auto result = integrator.integrate(integrand, {.neval = 1000000, .niter = 3, .verbosity = 1});
  std::cout << "integral = " << result.value() << " +/- " << result.error() << "\n";
  integrand.print_histogram();

  return 0;
}
