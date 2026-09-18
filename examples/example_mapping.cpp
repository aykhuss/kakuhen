#include "kakuhen/kakuhen.h"
#include <iostream>
#include <vector>

// bounds on the integration domain
constexpr double x_low = 1e-6; //0.5;
constexpr double x_upp = 1.0; //0.75;

int main() {
  using namespace kakuhen::integrator;

  auto func_naive = [](const Point<>& point) {
    const auto& x = point.x;  // shorthand
    if (x[0] < x_low || x[1] < x_low || x[0] > x_upp || x[1] > x_upp) return 0.0;
    return 1.0;
  };

  auto func_lin = [](const Point<>& point) {
    const auto& x = point.x;  // shorthand
    double wgt = 1.0;
    const double y0 = x_low + (x_upp - x_low) * x[0];
    wgt *= (x_upp - x_low);
    const double y1 = x_low + (x_upp - x_low) * x[1];
    wgt *= (x_upp - x_low);
    return wgt;
  };

  auto func_log = [](const Point<>& point) {
    const auto& x = point.x;  // shorthand
    double wgt = 1.0;
    const double lnxx = std::log(x_upp / x_low);
    const double y0 = x_low * std::pow(x_upp / x_low, x[0]);
    wgt *= y0 * lnxx;
    const double y1 = x_low * std::pow(x_upp / x_low, x[1]);
    wgt *= y1 * lnxx;
    return wgt;
  };

  // set up the integrator
  //auto integrator = Vegas(2);
  auto integrator = Basin(2);
  //integrator.set_min_score(0.0);

  std::cout << "\n----------------------------\n";
  std::cout << "///   NAIVE ";
  std::cout << "\n----------------------------\n";
  integrator.integrate(func_naive, {.neval = 700, .niter = 5, .adapt = true});
  integrator.set_options({.frozen = true});
  integrator.save("save_nve");
  auto result_nve = integrator.integrate(func_naive, {.neval = 100000, .niter = 2});

  integrator.reset();  // prepare for the next integration

  std::cout << "\n----------------------------\n";
  std::cout << "///   LINEAR ";
  std::cout << "\n----------------------------\n";
  integrator.integrate(func_lin, {.neval = 700, .niter = 5, .adapt = true});
  integrator.set_options({.frozen = true});
  integrator.save("save_lin");
  auto result_lin = integrator.integrate(func_lin, {.neval = 100000, .niter = 2});

  integrator.reset();  // prepare for the next integration

  std::cout << "\n----------------------------\n";
  std::cout << "///   LOGARITHMIC ";
  std::cout << "\n----------------------------\n";
  integrator.integrate(func_log, {.neval = 700, .niter = 5, .adapt = true});
  integrator.set_options({.frozen = true});
  integrator.save("save_log");
  auto result_log = integrator.integrate(func_log, {.neval = 100000, .niter = 2});

  std::cout << "\n----------------------------\n";
  std::cout << "\n\n";
  std::cout << "nve = " << result_nve.value() << " +/- " << result_nve.error();
  std::cout << " (ntotal=" << result_nve.count() << ", chi2/dof=" << result_nve.chi2dof() << ")\n";
  std::cout << "\n";
  std::cout << "lin = " << result_lin.value() << " +/- " << result_lin.error();
  std::cout << " (ntotal=" << result_lin.count() << ", chi2/dof=" << result_lin.chi2dof() << ")\n";
  std::cout << "\n";
  std::cout << "log = " << result_log.value() << " +/- " << result_log.error();
  std::cout << " (ntotal=" << result_log.count() << ", chi2/dof=" << result_log.chi2dof() << ")\n";

  return 0;
}
