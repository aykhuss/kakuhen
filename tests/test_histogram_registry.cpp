#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <array>
#include "kakuhen/histogram/axis_view.h" // New axis implementation
#include "kakuhen/histogram/histogram_registry.h"

using namespace kakuhen::histogram;
using Catch::Approx;

TEST_CASE("HistogramRegistry booking and filling", "[HistogramRegistry]") {
  HistogramRegistry<> registry;

  // 1. Book Histograms (No Axis)
  auto h1_id = registry.book("hist_pt", 10, 1);
  auto h2_id = registry.book("hist_eta", 5, 2); // 5 bins, 2 values each

  REQUIRE(registry.data().size() == 20); // 10 + (5*2) = 20
  REQUIRE(registry.ids().size() == 2);
  REQUIRE(registry.get_name(h1_id) == "hist_pt");

  // 2. Create Buffer
  auto buffer = registry.create_buffer();

  // 3. Fill
  std::vector<double> v1 = {100.0};
  registry.fill(buffer, h1_id, static_cast<uint32_t>(0), std::span{v1}); // Global idx 0

  std::vector<double> v2 = {0.5, 0.6};
  registry.fill(buffer, h2_id, static_cast<uint32_t>(0), std::span{v2}); // Global idx 10, 11 (start of h2)

  // Test scalar fill overload
  // Cast index to uint32_t to resolve ambiguity with (id, x, val) overload
  registry.fill(buffer, h1_id, static_cast<uint32_t>(1), 200.0); // Global idx 1

  // 4. Flush
  registry.flush(buffer);

  // 5. Verify
  REQUIRE(registry.data().bins()[0].weight() == Approx(100.0));
  REQUIRE(registry.data().bins()[1].weight() == Approx(200.0));
  REQUIRE(registry.data().bins()[10].weight() == Approx(0.5));
  REQUIRE(registry.data().bins()[11].weight() == Approx(0.6));
}

TEST_CASE("HistogramRegistry Axis Integration", "[HistogramRegistry]") {
  HistogramRegistry<> registry;

  // 1. Create Axes using Registry's create_axis helper
  auto u_id = registry.create_axis<UniformAxis<double, uint32_t>>(10, 0.0, 100.0);
  auto v_id = registry.create_axis<VariableAxis<double, uint32_t>>({0.0, 10.0, 100.0});

  // 2. Book Histograms with Axes via IDs
  auto h_u = registry.book("h_uniform", u_id);
  auto h_v = registry.book("h_variable", v_id);

  auto buffer = registry.create_buffer();

  // 3. Fill using x-values
  registry.fill(buffer, h_u, 5.0, 1.0);   // bin 0 [0,10) -> index 1
  registry.fill(buffer, h_u, 15.0, 2.0);  // bin 1 [10,20) -> index 2
  
  registry.fill(buffer, h_v, 5.0, 1.0);   // bin 0 [0,10) -> index 13
  registry.fill(buffer, h_v, 50.0, 2.0);  // bin 1 [10,100) -> index 14

  // --- Multi-valued Fill with x-coordinate ---
  auto h_mv = registry.book("h_multivalue_axis", u_id, 2); // reuse u_id, 2 values per bin
  std::array<double, 2> mv_vals = {10.0, 20.0};
  registry.fill(buffer, h_mv, 5.0, std::span{mv_vals}); // [0,10) -> bin 1

  registry.flush(buffer);

  // 4. Verify
  // h_u starts at 0. Size 12 (10 + 2).
  // bin 0: underflow
  // bin 1: [0, 10) -> 5.0 -> weight 1.0
  // bin 2: [10, 20) -> 15.0 -> weight 2.0
  REQUIRE(registry.data().bins()[1].weight() == Approx(1.0));
  REQUIRE(registry.data().bins()[2].weight() == Approx(2.0));

  // h_v starts at 12. Size 4 (2 + 2).
  // bin 0 (global 12): underflow
  // bin 1 (global 13): [0, 10) -> 5.0 -> weight 1.0
  // bin 2 (global 14): [10, 100) -> 50.0 -> weight 2.0
  REQUIRE(registry.data().bins()[13].weight() == Approx(1.0));
  REQUIRE(registry.data().bins()[14].weight() == Approx(2.0));

  // h_mv starts after h_v. 12 + 4 = 16.
  // bin 1 of h_mv is at global 16 + 1*2 = 18.
  // values were 10.0, 20.0.
  REQUIRE(registry.data().bins()[18].weight() == Approx(10.0));
  REQUIRE(registry.data().bins()[19].weight() == Approx(20.0));
}

TEST_CASE("HistogramRegistry name lookup", "[HistogramRegistry]") {
    HistogramRegistry<> registry;
    auto id = registry.book("my_hist", 10);
    
    REQUIRE_NOTHROW(registry.get_id("my_hist"));
    REQUIRE(registry.get_id("my_hist") == id);
    REQUIRE_THROWS_AS(registry.get_id("non_existent"), std::runtime_error);
}

TEST_CASE("HistogramRegistry accessors", "[HistogramRegistry]") {
  HistogramRegistry<> registry;
  auto h_id = registry.book("h", 10);
  auto buffer = registry.create_buffer();

  // --- Event 1 ---
  // Fill bin 2 with weight 2.0
  registry.fill(buffer, h_id, static_cast<uint32_t>(2), 2.0);
  registry.flush(buffer);

  // N=1
  // Mean = 2.0 / 1 = 2.0
  // Variance = 0 (N<=1)
  REQUIRE(registry.value(h_id, 2) == Approx(2.0));
  REQUIRE(registry.error(h_id, 2) == Approx(0.0));
  REQUIRE(registry.variance(h_id, 2) == Approx(0.0));

  // --- Event 2 ---
  // Fill bin 2 with weight 6.0
  registry.fill(buffer, h_id, static_cast<uint32_t>(2), 6.0);
  registry.flush(buffer);

  // N=2
  // Sum(w) = 2 + 6 = 8
  // Sum(w^2) = 4 + 36 = 40
  // Mean = 8 / 2 = 4.0
  // Variance = (40/2 - 4*4) / (2-1) = (20 - 16) / 1 = 4.0
  // Error = sqrt(4.0) = 2.0
  REQUIRE(registry.value(h_id, 2) == Approx(4.0));
  REQUIRE(registry.error(h_id, 2) == Approx(2.0));
  REQUIRE(registry.variance(h_id, 2) == Approx(4.0));

  // Check bin object access (raw sums)
  const auto& bin = registry.get_bin(h_id, 2);
  REQUIRE(bin.weight() == Approx(8.0));

  // Check out of bounds
  REQUIRE_THROWS_AS(registry.get_bin(h_id, 10), std::out_of_range);
}
