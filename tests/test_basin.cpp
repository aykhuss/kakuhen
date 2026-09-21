#include "kakuhen/integrator/basin.h"
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace kakuhen::integrator;

static auto test_integrand = [](const Point<>& point) {
  const auto& x = point.x;  // shorthand
  return (x[0] + x[1]) / (1. + x[0] - x[1]);
};

TEST_CASE("Basin ndim", "[basin]") {
  auto bas = Basin(3);
  REQUIRE(bas.ndim() == 3);
}

TEST_CASE("write/load state and data", "[basin]") {
  std::stringstream ss;

  auto veg = Basin(2);
  veg.set_options({.verbosity = 0});

  /// quick adaption:  save state
  veg.integrate(test_integrand, {.neval = 1000, .niter = 10, .adapt = true});
  veg.write_state_stream(ss);

  /// 2nd vegas to load state into
  auto veg_alt = Basin(2);
  veg_alt.set_options({.verbosity = 0});
  veg_alt.read_state_stream(ss);
  REQUIRE(veg.hash().value() == veg_alt.hash().value());

  /// another warmup:  no adaption; save data
  ss.str("");  // clear content
  ss.clear();  // clear flags
  veg.integrate(test_integrand, {.neval = 1000, .niter = 10, .adapt = false});
  veg.write_data_stream(ss);
  veg.adapt();

  /// read data & adapt separately
  veg_alt.read_data_stream(ss);
  veg_alt.adapt();
  REQUIRE(veg.hash().value() == veg_alt.hash().value());
}

TEST_CASE("Basin rejects invalid adaptation parameters", "[basin]") {
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  constexpr double inf = std::numeric_limits<double>::infinity();
  auto bas = Basin(2);
  for (const double alpha : {-1e-3, nan, inf}) {
    REQUIRE_THROWS_AS(bas.set_alpha(alpha), std::invalid_argument);
  }
  for (const double weight : {0.5, nan, inf}) {
    REQUIRE_THROWS_AS(bas.set_weight_smooth(weight), std::invalid_argument);
  }
  for (const double score : {-1e-3, 1.0, nan}) {
    REQUIRE_THROWS_AS(bas.set_min_score(score), std::invalid_argument);
  }
  REQUIRE(bas.alpha() == 0.75);
  REQUIRE(bas.weight_smooth() == 3.0);
  REQUIRE(bas.min_score() == 0.05);
  bas.set_alpha(0.0);
  bas.set_weight_smooth(1.0);
  bas.set_min_score(0.0);
  REQUIRE(bas.alpha() == 0.0);
  REQUIRE(bas.weight_smooth() == 1.0);
  REQUIRE(bas.min_score() == 0.0);
}
