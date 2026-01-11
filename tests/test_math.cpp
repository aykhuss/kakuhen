#include "kakuhen/util/math.h"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace kakuhen::util::math;

TEST_CASE("Math utilities: abs", "[math]") {
  REQUIRE(abs(5) == 5);
  REQUIRE(abs(-5) == 5);
  REQUIRE(abs(0) == 0);
  REQUIRE(abs(-3.14) == 3.14);
}

TEST_CASE("Math utilities: sq", "[math]") {
  REQUIRE(sq(2) == 4);
  REQUIRE(sq(-2) == 4);
  REQUIRE(sq(0) == 0);
  REQUIRE(sq(1.5) == 2.25);
}

TEST_CASE("Math utilities: sgn", "[math]") {
  REQUIRE(sgn(10) == 1);
  REQUIRE(sgn(-5) == -1);
  REQUIRE(sgn(0) == 0);
  REQUIRE(sgn(0.001) == 1);
}

TEST_CASE("Math utilities: ipow", "[math]") {
  SECTION("Integer base") {
    REQUIRE(ipow(2, 0) == 1);
    REQUIRE(ipow(2, 1) == 2);
    REQUIRE(ipow(2, 3) == 8);
    REQUIRE(ipow(2, 10) == 1024);
    REQUIRE(ipow(3, 3) == 27);
    REQUIRE(ipow(-2, 3) == -8);  // Negative base, odd exp
    REQUIRE(ipow(-2, 2) == 4);   // Negative base, even exp

    // Negative exponents for integers (should be 0, 1, or -1)
    REQUIRE(ipow(2, -2) == 0);
    REQUIRE(ipow(1, -5) == 1);
    REQUIRE(ipow(-1, -2) == 1);
    REQUIRE(ipow(-1, -3) == -1);
  }

  SECTION("Floating point base") {
    REQUIRE(ipow(2.0, 3) == 8.0);
    REQUIRE(ipow(2.0, -1) == 0.5);
    REQUIRE(ipow(2.0, -2) == 0.25);
  }
}

TEST_CASE("Math utilities: nearly_equal", "[math]") {
  REQUIRE(nearly_equal(1.0, 1.0));
  REQUIRE(nearly_equal(1.0, 1.0000000000000001));  // small diff
  REQUIRE_FALSE(nearly_equal(1.0, 1.0001));

  // Zero handling
  REQUIRE(nearly_equal(0.0, 0.0));
  REQUIRE(nearly_equal(0.0, -0.0));

  // NaN/Inf
  double nan = std::numeric_limits<double>::quiet_NaN();
  double inf = std::numeric_limits<double>::infinity();
  REQUIRE_FALSE(nearly_equal(nan, nan));
  REQUIRE_FALSE(nearly_equal(1.0, nan));
  REQUIRE(nearly_equal(inf, inf));
  REQUIRE_FALSE(nearly_equal(inf, -inf));
  REQUIRE_FALSE(nearly_equal(1.0, inf));
}
