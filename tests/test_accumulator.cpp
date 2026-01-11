#include "kakuhen/integrator/grid_accumulator.h"
#include "kakuhen/integrator/integral_accumulator.h"
#include "kakuhen/util/accumulator.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <sstream>

#include "catch2/catch_approx.hpp"

using Catch::Matchers::Message;

TEST_CASE("Accumulator algorithms coverage", "[accumulator]") {
  using kakuhen::util::accumulator::AccumAlgo;
  using kakuhen::util::accumulator::Accumulator;

  // New API: Accumulator<Type, Algo>
  auto accNaive = Accumulator<double, AccumAlgo::Naive>();
  auto accKahan = Accumulator<double, AccumAlgo::Kahan>(1.);
  auto accNeumaier = Accumulator<double, AccumAlgo::Neumaier>(2.3);
  auto accTwoSum = Accumulator<double, AccumAlgo::TwoSum>(42e-2);
  auto accDefault = Accumulator<double>();  // Should default to TwoSum

  SECTION("Naive Accumulator") {
    accNaive.add(1.0);
    REQUIRE(accNaive.result() == Catch::Approx(1.0));
    // feature dropped: accNaive = 5.0;  // scalar assignment
    // REQUIRE(accNaive.result() == Catch::Approx(5.0));
  }

  SECTION("Kahan Accumulator") {
    accKahan.add(1.0);
    accKahan += 2.0;
    REQUIRE(accKahan.result() == Catch::Approx(4.0));
  }

  SECTION("Neumaier Accumulator") {
    accNeumaier += 3.0;
    double res = accNeumaier;
    REQUIRE(res == Catch::Approx(5.3));
  }

  SECTION("TwoSum Accumulator") {
    REQUIRE(accTwoSum.result() == Catch::Approx(0.42));
    accTwoSum.reset(1.1);
    accTwoSum += 2.3;
    REQUIRE(static_cast<double>(accTwoSum) == Catch::Approx(3.4));

    // // Test renormalize (specific to TwoSum)
    // accTwoSum.renormalize();
    // REQUIRE(accTwoSum.result() == Catch::Approx(3.4));
  }

  SECTION("Default Accumulator") {
    accDefault.add(10.0);
    REQUIRE(accDefault.result() == Catch::Approx(10.0));
  }
}

TEST_CASE("Accumulator serialization", "[accumulator]") {
  using kakuhen::util::accumulator::AccumAlgo;
  using kakuhen::util::accumulator::Accumulator;
  std::stringstream ss;

  auto accTwoSum = Accumulator<double, AccumAlgo::TwoSum>(42e-2);
  accTwoSum += 2.3;
  REQUIRE(accTwoSum.result() == Catch::Approx(2.72));
  accTwoSum.serialize(ss);

  auto accTwoSum_read = Accumulator<double, AccumAlgo::TwoSum>();
  accTwoSum_read.deserialize(ss);

  REQUIRE(accTwoSum_read.result() == Catch::Approx(2.72));

  /// test type mismatch
  ss.str("");
  accTwoSum.serialize(ss, true);
  auto accTwoSum_mismatch = Accumulator<float, AccumAlgo::TwoSum>();
  REQUIRE_THROWS_MATCHES(accTwoSum_mismatch.deserialize(ss, true), std::runtime_error,
                         Message("type or size mismatch for typename T"));
}

TEST_CASE("IntegralAccumulator tests", "[integrator][accumulator]") {
  using kakuhen::integrator::IntegralAccumulator;
  using kakuhen::integrator::make_integral_accumulator;

  IntegralAccumulator<double, unsigned> intAcc;

  SECTION("Basic Accumulation") {
    intAcc.accumulate(1.0);  // f=1, f2=1, n=1
    intAcc.accumulate(3.0);  // f=4, f2=10, n=2

    REQUIRE(intAcc.count() == 2);
    REQUIRE(intAcc.value() == Catch::Approx(2.0));  // mean = 4/2

    // Variance: (10/2 - 2*2) / 1 = (5 - 4) = 1
    REQUIRE(intAcc.variance() == Catch::Approx(1.0));
    REQUIRE(intAcc.error() == Catch::Approx(1.0));
  }

  SECTION("Reset and make_integral_accumulator") {
    intAcc = make_integral_accumulator<double, unsigned>(2.0, 1.0, 2);
    // Constructed from mean=2, error=1, n=2 -> should match previous state
    REQUIRE(intAcc.count() == 2);
    REQUIRE(intAcc.value() == Catch::Approx(2.0));
    REQUIRE(intAcc.variance() == Catch::Approx(1.0));
  }

  SECTION("Accumulate from other") {
    IntegralAccumulator<double, unsigned> other;
    other.accumulate(5.0);
    intAcc.accumulate(other);
    REQUIRE(intAcc.value() == Catch::Approx(5.0));
  }
}

TEST_CASE("GridAccumulator tests", "[integrator][accumulator]") {
  using kakuhen::integrator::GridAccumulator;

  GridAccumulator<double, unsigned> gridAcc;

  SECTION("Accumulation") {
    gridAcc.accumulate(10.0);
    gridAcc.accumulate(20.0);

    REQUIRE(gridAcc.count() == 2);
    REQUIRE(gridAcc.value() == Catch::Approx(30.0));
  }

  SECTION("Reset") {
    gridAcc.accumulate(10.0);
    gridAcc.reset();
    REQUIRE(gridAcc.count() == 0);
    REQUIRE(gridAcc.value() == Catch::Approx(0.0));

    gridAcc.reset(5.0, 1);
    REQUIRE(gridAcc.count() == 1);
    REQUIRE(gridAcc.value() == Catch::Approx(5.0));
  }

  SECTION("Accumulate from other") {
    GridAccumulator<double, unsigned> other;
    other.accumulate(5.0);
    gridAcc.accumulate(other);
    REQUIRE(gridAcc.value() == Catch::Approx(5.0));
  }
}
