/// Generating exactly N unweighted events.
///
/// The generator only runs fixed trial budgets (`generate_trials`): a fixed
/// budget always terminates, and a completed run gives an unbiased
/// normalization. To end up with exactly `N` events instead:
///
///   1. size the budget from the predicted unweighting efficiency,
///   2. stop from the event callback once the N-th event arrives,
///   3. top up with further batches if a batch fell short.

#include "kakuhen/integrator/vegas_generator.h"
#include <cmath>
#include <iostream>
#include <numbers>

using namespace kakuhen::integrator;

int main() {
  using count_type = Point<>::count_type;

  /// a Gaussian peak in 2D; its integral over the unit square is ≈ π/50
  auto integrand = [](const Point<>& point) {
    const double dx = point.x[0] - 0.5;
    const double dy = point.x[1] - 0.5;
    return std::exp(-50.0 * (dx * dx + dy * dy));
  };

  /// adapt the grid, then freeze it for generation
  VegasGenerator<> gen(2, 32);
  gen.integrate(integrand,
                {.neval = 20000, .niter = 5, .adapt = true, .verbosity = 0, .progress_bar = false});
  gen.set_options({.frozen = true, .verbosity = 0});

  /// build the envelope from a sampled abs-integral estimate, which also
  /// feeds `predicted_efficiency()`
  gen.initialize_envelope(integrand, 10000);
  gen.optimize_envelope(integrand, 100000);

  /// the event sink: consume the event, then stop once we have `nevents`
  const count_type nevents = 100000;
  count_type n = 0;
  double weight_sum = 0.0;
  auto sink = [&](const Point<>& /* point */, double weight) {
    // ... write the event (point.x, weight) to a file or histogram here ...
    weight_sum += weight;
    return ++n < nevents ? EventSignal::NONE : EventSignal::STOP;
  };

  /// Accepted events per trial never exceed the predicted efficiency (up to
  /// its statistical error), and the event count fluctuates by ~√N: a small
  /// headroom makes a top-up rare. It is a choice, not a library policy.
  constexpr double headroom = 1.05;
  auto budget = [&](count_type missing, double efficiency) {
    return static_cast<count_type>(std::ceil(headroom * double(missing) / efficiency));
  };

  /// Every batch is finite, and the loop only continues after a batch that
  /// delivered at least one event, so it runs at most `nevents` times. A batch
  /// without any event means the integrand (almost) vanishes under the
  /// envelope; give up instead of spinning.
  VegasGenerator<>::gen_result_type result;
  double efficiency = gen.predicted_efficiency();
  count_type nbatches = 0;
  while (n < nevents) {
    const auto batch = gen.generate_trials(integrand, budget(nevents - n, efficiency), sink);
    result.accumulate(batch);
    ++nbatches;
    if (batch.n_events() == 0) {
      std::cerr << "no event in " << batch.n_trials() << " trials; giving up\n";
      break;
    }
    efficiency = result.efficiency();  // top-ups use the realized efficiency
  }

  std::cout << "events = " << n << " [" << to_string(result.status()) << "]"
            << " | trials = " << result.n_trials() << " | batches = " << nbatches
            << " | predicted eff = " << gen.predicted_efficiency()
            << " | realized eff = " << result.efficiency() << "\n";

  /// Events carry weights ±1 (rare overweights ±|f|/R); `normalization()` =
  /// V / n_trials turns them into a physical sample. Stopping on the N-th
  /// event biases this ratio by O(1/N), negligible at this N. For an exactly
  /// unbiased normalization, run a `generate_trials` budget to completion
  /// without stopping.
  std::cout << "integral from events = " << result.normalization() * weight_sum << " ("
            << result.value() << " +/- " << result.error() << ")"
            << " | expected ≈ " << std::numbers::pi / 50.0 << "\n";

  return 0;
}
