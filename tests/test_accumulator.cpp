#include "kakuhen/util/accumulator.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <sstream>

#include "catch2/catch_approx.hpp"

using Catch::Matchers::Message;

TEST_CASE("different accumulators & cover all routines", "[accumulator]") {
  using kakuhen::util::accumulator::AccumAlgo;
  using kakuhen::util::accumulator::Accumulator;

  auto accNaive = Accumulator<AccumAlgo::Naive, double>();
  auto accKahan = Accumulator<AccumAlgo::Kahan, double>(1.);
  auto accNeumaier = Accumulator<AccumAlgo::Neumaier, double>(2.3);
  auto accTwoSum = Accumulator<AccumAlgo::TwoSum, double>(42e-2);

  accNaive.add(1.0);
  REQUIRE(accNaive.result() == Catch::Approx(1.0));

  accKahan.add(1.0);
  accKahan += 2.0;
  REQUIRE(accKahan.result() == Catch::Approx(4.0));

  accNeumaier += 3.0;
  double res = accNeumaier;
  REQUIRE(res == Catch::Approx(5.3));

  REQUIRE(accTwoSum.result() == Catch::Approx(0.42));
  accTwoSum.reset(1.1);
  accTwoSum += 2.3;
  REQUIRE(static_cast<double>(accTwoSum) == Catch::Approx(3.4));
}

TEST_CASE("Accumulator serialization", "[accumulator]") {
  using kakuhen::util::accumulator::AccumAlgo;
  using kakuhen::util::accumulator::Accumulator;
  std::stringstream ss;

  auto accTwoSum = Accumulator<AccumAlgo::TwoSum, double>(42e-2);
  accTwoSum += 2.3;
  REQUIRE(accTwoSum.result() == Catch::Approx(2.72));
  accTwoSum.serialize(ss);

  auto accTwoSum_read = Accumulator<AccumAlgo::TwoSum, double>();
  accTwoSum_read.deserialize(ss);

  REQUIRE(accTwoSum_read.result() == Catch::Approx(2.72));

  /// test type mismatch
  ss.str("");
  accTwoSum.serialize(ss, true);
  auto accTwoSum_mismatch = Accumulator<AccumAlgo::TwoSum, float>();
  REQUIRE_THROWS_MATCHES(accTwoSum_mismatch.deserialize(ss, true), std::runtime_error,
                         Message("type or size mismatch for typename T"));
}

//@todo:  add tests for integralaccumulator and closure for make_integral_accumulator.
//@todo:  tests for grid_accumulator?
