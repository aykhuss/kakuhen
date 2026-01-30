#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "kakuhen/histogram/axis.h"
#include "kakuhen/histogram/histogram_registry.h"
#include <sstream>

using namespace kakuhen::histogram;
using Catch::Approx;

TEST_CASE("HistogramRegistry full state serialization", "[HistogramRegistry]") {
  HistogramRegistry<> registry;

  // 1. Setup State
  UniformAxis<> u_ax(10, 0.0, 100.0);
  VariableAxis<> v_ax({0.0, 10.0, 100.0});

  auto h_u = registry.book("h_uniform", 1, u_ax);
  auto h_v = registry.book("h_variable", 1, v_ax);
  auto h_n = registry.book("h_noaxis", 1, 5);

  auto buffer = registry.create_buffer();
  registry.fill(buffer, h_u, 1.0, 5.0);
  registry.fill(buffer, h_u, 2.0, 15.0);
  registry.fill(buffer, h_v, 3.0, 50.0);
  registry.fill_by_index(buffer, h_n, 4.0, static_cast<uint32_t>(2));
  registry.flush(buffer);

  // Capture weights before serialization
  double w_u1 = registry.value(h_u, 1);
  double w_u2 = registry.value(h_u, 2);
  double w_v = registry.value(h_v, 2);
  double w_n = registry.value(h_n, 2);

  // 2. Serialize
  std::stringstream ss;
  registry.serialize(ss);

  // 3. Deserialize into a new registry
  HistogramRegistry<> back_registry;
  back_registry.deserialize(ss);

  // 4. Verify Metadata
  REQUIRE(back_registry.ids().size() == 3);
  REQUIRE(back_registry.get_name(h_u) == "h_uniform");
  REQUIRE(back_registry.get_name(h_v) == "h_variable");
  REQUIRE(back_registry.get_name(h_n) == "h_noaxis");

  // 5. Verify Data
  REQUIRE(back_registry.value(h_u, 1) == Approx(w_u1));
  REQUIRE(back_registry.value(h_u, 2) == Approx(w_u2));
  REQUIRE(back_registry.value(h_v, 2) == Approx(w_v));
  REQUIRE(back_registry.value(h_n, 2) == Approx(w_n));

  // 6. Verify continuation of filling
  auto back_buffer = back_registry.create_buffer();
  back_registry.fill(back_buffer, h_u, 10.0, 5.0);
  back_registry.flush(back_buffer);

  // Original N was 1. New N should be 2.
  REQUIRE(back_registry.data().count() == 2);

  // Original sum was w_u1 * N_orig. New sum is (w_u1 * N_orig) + 10.0.
  // New mean is (w_u1 * N_orig + 10.0) / (N_orig + 1).
  double N_orig = static_cast<double>(registry.data().count());
  REQUIRE(N_orig == 1.0);
  double expected_mean = (w_u1 * N_orig + 10.0) / (N_orig + 1.0);
  REQUIRE(back_registry.value(h_u, 1) == Approx(expected_mean));
}

TEST_CASE("BinAccumulator collapsed serialization", "[BinAccumulator]") {
    BinAccumulator<double> bin;
    bin.accumulate(1.0);
    bin.accumulate(2.0);

    REQUIRE(bin.weight() == Approx(3.0));
    REQUIRE(bin.weight_sq() == Approx(5.0)); // 1^2 + 2^2 = 5

    std::stringstream ss;
    bin.serialize(ss);

    BinAccumulator<double> back_bin;
    back_bin.deserialize(ss);

    REQUIRE(back_bin.weight() == Approx(3.0));
    REQUIRE(back_bin.weight_sq() == Approx(5.0));

    // Verify we can continue accumulating
    back_bin.accumulate(3.0);
    REQUIRE(back_bin.weight() == Approx(6.0));
    REQUIRE(back_bin.weight_sq() == Approx(14.0)); // 5 + 3^2 = 14
}
