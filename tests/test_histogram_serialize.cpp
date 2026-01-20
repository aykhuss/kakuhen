#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "kakuhen/histogram/histogram_registry.h"
#include <sstream>

using namespace kakuhen::histogram;
using Catch::Approx;

TEST_CASE("HistogramRegistry full state serialization", "[HistogramRegistry]") {
  HistogramRegistry<> registry;

  // 1. Setup State
  auto u_id = registry.create_axis<UniformAxis<double, uint32_t>>(10, 0.0, 100.0);
  auto v_id = registry.create_axis<VariableAxis<double, uint32_t>>({0.0, 10.0, 100.0});

  auto h_u = registry.book("h_uniform", u_id);
  auto h_v = registry.book("h_variable", v_id);
  auto h_n = registry.book("h_noaxis", 5);

  auto buffer = registry.create_buffer();
  registry.fill(buffer, h_u, 5.0, 1.0);
  registry.fill(buffer, h_u, 15.0, 2.0);
  registry.fill(buffer, h_v, 50.0, 3.0);
  registry.fill(buffer, h_n, static_cast<uint32_t>(2), 4.0);
  registry.flush(buffer);

  // Capture weights before serialization
  double w_u1 = registry.data().bins()[1].weight();
  double w_u2 = registry.data().bins()[2].weight();
  double w_v = registry.data().bins()[14].weight(); // offset 12 + 2
  double w_n = registry.data().bins()[18].weight(); // offset 12 + 4 + 2

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
  REQUIRE(back_registry.data().bins()[1].weight() == Approx(w_u1));
  REQUIRE(back_registry.data().bins()[2].weight() == Approx(w_u2));
  REQUIRE(back_registry.data().bins()[14].weight() == Approx(w_v));
  REQUIRE(back_registry.data().bins()[18].weight() == Approx(w_n));

  // 6. Verify continuation of filling
  auto back_buffer = back_registry.create_buffer();
  back_registry.fill(back_buffer, h_u, 5.0, 10.0);
  back_registry.flush(back_buffer);

  REQUIRE(back_registry.data().bins()[1].weight() == Approx(w_u1 + 10.0));
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
