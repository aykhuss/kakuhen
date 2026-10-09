/// Train a generation envelope from independent workers using mergeable .khe files.
/// The worker loop runs sequentially here; each iteration can be a separate process.

#include "kakuhen/integrator/basin_generator.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main() {
  using namespace kakuhen::integrator;
  using Generator = BasinGenerator<>;

  // Integral over [0, 1]^2 = 2/3.
  auto integrand = [](const Point<>& point) {
    const auto& x = point.x;
    return x[0] * x[0] + x[1] * x[1];
  };

  const std::filesystem::path output_dir = "distributed_generator";
  std::filesystem::create_directories(output_dir);
  const auto initial_checkpoint = output_dir / "generator.khs";
  const auto trained_checkpoint = output_dir / "generator_trained.khs";

  // 1. Prepare a common sampling map and initial envelope for all workers.
  Generator generator(2, 8, 16);
  generator.set_seed(42);
  generator.integrate(integrand, {.neval = 20000, .niter = 5, .adapt = true, .verbosity = 0});
  generator.set_options({.frozen = true, .verbosity = 0});
  generator.initialize_envelope(integrand, 10000);
  generator.save(initial_checkpoint);

  // 2. Each worker loads the same checkpoint, but uses a distinct RNG seed.
  // Unlike .khd collection, envelope collection requires a frozen grid.
  std::vector<std::filesystem::path> envelope_files;
  for (Generator::seed_type seed = 101; seed <= 104; ++seed) {
    Generator worker(2);
    worker.load(initial_checkpoint);
    worker.set_options({.frozen = true, .verbosity = 0});  // Options are not serialized.
    worker.set_seed(seed);

    // Defer envelope updates until the coordinator replays these observations.
    const auto collected = worker.collect_envelope(integrand, 50000);
    const auto envelope_file = output_dir / ("worker_" + std::to_string(seed) + ".khe");
    worker.save_envelope(envelope_file);
    envelope_files.push_back(envelope_file);
    std::cout << "Worker " << seed << ": " << collected.n_evaluations() << " evaluations, "
              << collected.n_violations() << " records -> " << envelope_file << "\n";

    // Saving leaves the batch pending. If this worker collects another export,
    // call clear_envelope_data() first so batches do not overlap.
  }

  // 3. Merge each worker file once, in a fixed order. Do not adapt the grid:
  // every file refers to the sampling map in the common checkpoint.
  Generator combined(2);
  combined.load(initial_checkpoint);
  combined.set_options({.frozen = true, .verbosity = 0});
  for (const auto& envelope_file : envelope_files) {
    const auto merged = combined.merge_envelope(envelope_file);
    std::cout << "Merged " << envelope_file << ": " << merged.n_raised() << " envelope raises\n";
  }
  // Merging already replays incoming observations; no adapt_envelope() is needed.
  combined.save(trained_checkpoint);

  // 4. Distribute the trained checkpoint to production workers and reseed each.
  Generator production(2);
  production.load(trained_checkpoint);
  production.set_options({.frozen = true, .verbosity = 0});
  production.set_seed(1001);
  double weight_sum = 0.0;
  const auto events =
      production.generate_trials(integrand, 100000, [&](const Point<>& /*point*/, double weight) {
        // Store (point.x, weight), or fill histograms here. Preserve overweight
        // corrections even for a positive integrand like this one.
        weight_sum += weight;
      });

  std::cout << "Generated " << events.n_events() << " events in " << events.n_trials()
            << " trials (" << events.n_overweight() << " overweight)\n";
  std::cout << "Integral from events = " << events.normalization() * weight_sum << " +/- "
            << events.error() << " (expected " << 2.0 / 3.0 << ")\n";
  return 0;
}
