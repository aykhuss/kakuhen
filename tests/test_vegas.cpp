#include "kakuhen/integrator/vegas.h"
#include <catch2/catch_test_macros.hpp>
#include <sstream>

using namespace kakuhen::integrator;

static auto func = [](const Point<>& point) {
  const auto& x = point.x;  // shorthand
  return (x[0] + x[1]) / (1. + x[0] - x[1]);
};

TEST_CASE("write/load state and data", "[vegas]") {
  std::stringstream ss;

  auto veg = Vegas(2);
  veg.set_options({.verbosity = 0});

  /// quick adaption:  save state
  veg.integrate(func, {.neval = 1000, .niter = 10, .adapt = true});
  veg.write_state_stream(ss);

  /// 2nd vegas to load state into
  auto veg_alt = Vegas(2);
  veg_alt.set_options({.verbosity = 0});
  veg_alt.read_state_stream(ss);
  REQUIRE(veg.hash().value() == veg_alt.hash().value());

  /// another warmup:  no adaption; save data
  ss.clear();
  veg.integrate(func, {.neval = 1000, .niter = 10, .adapt = false});
  veg.write_data_stream(ss);
  veg.adapt();

  /// read data & adapt separately
  veg_alt.read_data_stream(ss);
  veg_alt.adapt();
  REQUIRE(veg.hash().value() == veg_alt.hash().value());
}

TEST_CASE("write/load RNG state", "[vegas]") {
  std::stringstream ss_grid, ss_rng;

  auto veg = Vegas(2);
  veg.set_options({.verbosity = 0});
  veg.set_seed(42);

  /// quick adaption:  save state
  veg.integrate(func, {.neval = 1000, .niter = 10, .adapt = true});
  veg.write_state_stream(ss_grid);
  veg.write_rng_state_stream(ss_rng);

  /// synchronize full state of veg
  auto veg2 = Vegas(2);
  veg2.set_options({.verbosity = 0});
  veg2.read_state_stream(ss_grid);
  veg2.read_rng_state_stream(ss_rng);

  auto res1 = veg.integrate(func, {.neval = 1000, .niter = 10, .adapt = true});
  auto res2 = veg2.integrate(func, {.neval = 1000, .niter = 10, .adapt = true});

  REQUIRE(veg.hash().value() == veg2.hash().value());
  REQUIRE(res1.value() == res2.value());
  REQUIRE(res1.error() == res2.error());
}
