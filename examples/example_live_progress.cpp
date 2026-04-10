#include <kakuhen/integrator/vegas.h>
#include <kakuhen/util/progress_bar.h>

#include "kakuhen/integrator/progress.h"
#include <cmath>
#include <iostream>

// A slow integrand to demonstrate the progress bar
double slow_integrand(const kakuhen::integrator::Point<>& p) {
  // Simulate some computation
  double sum = 0;
  for (int i = 0; i < 5000; ++i) {
    sum += std::sin(p.x[0] * i) * std::cos(p.x[1] * i);
  }
  return std::exp(-p.x[0] * p.x[0] - p.x[1] * p.x[1]) + sum * 1e-10;
}

int main() {
  using namespace kakuhen::integrator;

  Vegas integrator(2);  // 2D integration
  integrator.set_options({.progress_step = 0.01});

  // Create a progress bar (50 char width)
  kakuhen::util::ProgressBar bar(50);

  // Progress callback that updates the bar
  auto progress_cb = [&bar](const ProgressEvent<double, uint64_t>& ev) {
    if (ev.kind == ProgressEventKind::ITER_START ) {
      bar.reset();
    }
    bar.update(ev.fraction, "Integrating");
    return EventSignal::NONE;
  };

  std::cout << "Running Vegas integration with live progress bar...\n\n";

  // Run integration with progress callback
  auto result = integrator.integrate(
      slow_integrand,
      {.neval = 100000, .niter = 2, .verbosity = 1},
      progress_cb);

  std::cout << "\nResult: " << result.value() << " +/- " << result.error() << '\n';
  std::cout << "chi2/dof: " << result.chi2dof() << '\n';

  return 0;
}
