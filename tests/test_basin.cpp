#include "kakuhen/integrator/basin.h"
#include <catch2/catch_test_macros.hpp>
#include <sstream>

using namespace kakuhen::integrator;

auto func = [](const Point<>& point) {
  const auto& x = point.x;  // shorthand
  return (x[0] + x[1]) / (1. + x[0] - x[1]);
};

TEST_CASE("write/load state and data", "[basin]") {
  std::stringstream ss;

  auto veg = Basin(2);
  veg.set_options({.verbosity = 0});

  /// quick adaption:  save state
  veg.integrate(func, {.neval = 1000, .niter = 10, .adapt = true});
  veg.write_state_stream(ss);

  /// 2nd vegas to load state into
  auto veg_alt = Basin(2);
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
