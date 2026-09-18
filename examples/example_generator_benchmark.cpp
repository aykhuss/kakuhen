/// Compare the unweighting efficiency of `VegasGenerator` and `BasinGenerator`
/// on sample integrands.
///
/// Both generators run through the identical workflow:
///   1. adaptive warmup integration,
///   2. freeze the grid + one frozen production run,
///   3. seed the envelope from the production result (`initialize_envelope`),
///   4. raise the envelope until a pass records no violation (`optimize_envelope`),
///   5. generate a fixed number of trials and compare predicted vs. realized
///      efficiency, and the event-sample integral against the production run.
///
/// Step 4 is convergence-driven on purpose: a fixed small pass budget leaves the
/// tighter BASIN envelope under-converged. Its residual violations sit in the
/// wide outer conditional cells, where |f| peaks in a thin corner that uniform
/// raising samples rarely hit, and they show up as overweight events.
///
/// The takeaway: on integrands with inter-dimensional structure (ridges,
/// multiple peaks) the BASIN envelope mirrors the learned conditional sampling
/// order and reaches a substantially higher unweighting efficiency, while on
/// truly separable integrands the two models perform identically.
///
/// Both generators are configured with the same envelope resolution per
/// dimension: VEGAS `ndiv = 128` and BASIN `ndiv1 x ndiv2 = 8 x 16 = 128`.

#include "kakuhen/integrator/basin_generator.h"
#include "kakuhen/integrator/vegas_generator.h"
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace kakuhen::integrator;

namespace {

struct BenchmarkCase {
  std::string name;
  Point<>::size_type ndim;
  std::function<double(const Point<>&)> integrand;
};

/// run the full generation workflow and print the efficiency report
template <typename GEN>
void run(const char* label, GEN&& gen, const BenchmarkCase& bc) {
  gen.integrate(bc.integrand,
                {.neval = 50000, .niter = 7, .adapt = true, .verbosity = 0, .progress_bar = false});
  gen.set_options({.frozen = true, .verbosity = 0});
  auto prod = gen.integrate(bc.integrand,
                            {.neval = 50000, .niter = 1, .verbosity = 0, .progress_bar = false});

  gen.initialize_envelope(std::abs(prod.value()));
  // raise until a pass is violation-free (target rate 0), capped at 30 passes
  auto env = gen.optimize_envelope(bc.integrand, 100000, 30, 0.0);

  // fixed trial budget: unbiased integral estimate from the event sample
  auto sink = [](const Point<>&, double) {};
  auto res = gen.generate_trials(bc.integrand, 200000, sink);

  std::cout << "  " << std::left << std::setw(6) << label << std::right << std::fixed
            << std::setprecision(4) << "  predicted eff = " << env.efficiency()
            << "  realized eff = " << res.efficiency() << "  overweight = " << res.n_overweight()
            << " (max " << std::setprecision(2) << res.max_overweight() << ")\n"
            << std::scientific << std::setprecision(5) << "          integral: production = "
            << prod.value() << " +- " << std::setprecision(2) << prod.error()
            << "   events = " << std::setprecision(5) << res.value() << " +- "
            << std::setprecision(2) << res.error() << "\n"
            << std::defaultfloat;
}

}  // namespace

int main() {
  const std::vector<BenchmarkCase> cases = {
      {"separable peaks 3D (example_generator integrand)", 3,
       [](const Point<>& p) {
         const double r1[2] = {0.15, 0.2};
         const double r2[2] = {0.65, 0.5};
         double d1 = 0.;
         double d2 = 0.;
         for (std::size_t i = 0; i < 2; ++i) {
           d1 += (p.x[i] - r1[i]) * (p.x[i] - r1[i]);
           d2 += (p.x[i] - r2[i]) * (p.x[i] - r2[i]);
         }
         return 1e3 * std::exp(-50 * d1) + 7e2 * std::exp(-20 * d2);
       }},
      {"ridge 2D: exp(-100 (x0-x1)^2)", 2,
       [](const Point<>& p) {
         const double d = p.x[0] - p.x[1];
         return std::exp(-100.0 * d * d);
       }},
      {"peak x ridge 3D (diagonal structure)", 3,
       [](const Point<>& p) {
         const double r1[2] = {0.15, 0.2};
         const double r2[2] = {0.65, 0.5};
         double d1 = 0.;
         double d2 = 0.;
         for (std::size_t i = 0; i < 2; ++i) {
           d1 += (p.x[i] - r1[i]) * (p.x[i] - r1[i]);
           d2 += (p.x[i] - r2[i]) * (p.x[i] - r2[i]);
         }
         const double fval = 1e3 * std::exp(-50 * d1) + 7e2 * std::exp(-20 * d2);
         const double od = std::fabs(p.x[1] - p.x[2]);
         return fval * std::exp(-20. * od * od);
       }},
      {"separable poly 2D: 4 x0 x1", 2, [](const Point<>& p) { return 4.0 * p.x[0] * p.x[1]; }},
      {"tilted ridge 2D: exp(-100 (x0+x1-1)^2)", 2,
       [](const Point<>& p) {
         const double d = p.x[0] + p.x[1] - 1.0;
         return std::exp(-100.0 * d * d);
       }},
  };

  for (const auto& bc : cases) {
    std::cout << bc.name << "\n";
    run("vegas", VegasGenerator<>(bc.ndim), bc);  // ndiv = 128
    BasinGenerator<> basin(bc.ndim);              // ndiv0 = 8 * 16 = 128
    run("basin", basin, bc);
    std::cout << "         basin sampling: " << (bc.ndim - basin.nblocks())
              << " conditional dimension(s) out of " << bc.ndim << "\n\n";
  }

  return 0;
}
