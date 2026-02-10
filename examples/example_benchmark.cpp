#include "kakuhen/kakuhen.h"
#include "kakuhen/util/math.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
  // load the namespace for convenience
  using namespace kakuhen::integrator;

  const std::vector<double> r1(8, 0.23);
  const std::vector<double> r2(8, 0.39);
  const std::vector<double> r3(8, 0.74);

  auto func = [](const Point<>& point) {
    assert(point.ndim == 8);
    const auto& x = point.x;  // shorthand
    double dr1 = 0;
    double dr2 = 0;
    double dr3 = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      dr1 += kakuhen::util::math::ipow(x[i] - 0.23, 2);
      dr2 += kakuhen::util::math::ipow(x[i] - 0.39, 2);
      dr3 += kakuhen::util::math::ipow(x[i] - 0.74, 2);
    }
    return std::exp(-50. * std::sqrt(dr1)) +
           std::exp(-50. * std::sqrt(dr2)) +
           std::exp(-50. * std::sqrt(dr3));
  };

  // --- Basin Integration ---
  auto integrator_basin = Basin(8, 16, 32);
  std::cout << "\n--- Profiling Basin Algorithm ---\n";
  auto start = std::chrono::high_resolution_clock::now();
  integrator_basin.integrate(func, {.neval = 100000, .niter = 10, .adapt = true});
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "Basin Warmup time: " << diff.count() << " s\n";

  start = std::chrono::high_resolution_clock::now();
  auto result_basin = integrator_basin.integrate(func, {.neval = 100000, .niter = 20, .adapt = true});
  end = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "Basin Production time: " << diff.count() << " s\n";
  std::cout << "Basin Result: " << result_basin.value() << " +/- " << result_basin.error() << "\n";

  // --- Vegas Integration ---
  auto integrator_vegas = Vegas(8, 512);
  std::cout << "\n--- Profiling Vegas Algorithm ---\n";
  start = std::chrono::high_resolution_clock::now();
  integrator_vegas.integrate(func, {.neval = 100000, .niter = 10, .adapt = true});
  end = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "Vegas Warmup time: " << diff.count() << " s\n";

  start = std::chrono::high_resolution_clock::now();
  auto result_vegas = integrator_vegas.integrate(func, {.neval = 100000, .niter = 20, .adapt = true});
  end = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "Vegas Production time: " << diff.count() << " s\n";
  std::cout << "Vegas Result: " << result_vegas.value() << " +/- " << result_vegas.error() << "\n";

  return 0;
}
