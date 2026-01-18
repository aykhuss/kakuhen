#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
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
  registry.fill(buffer, h1_id, 0, v1); // Global idx 0

  std::vector<double> v2 = {0.5, 0.6};
  registry.fill(buffer, h2_id, 0, v2); // Global idx 10, 11 (start of h2)

  // Test scalar fill overload
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

  // 1. Create Axes using Registry's data
  UniformAxis<double, uint32_t> u_axis(registry.axis_data(), 10, 0.0, 100.0);
  VariableAxis<double, uint32_t> v_axis(registry.axis_data(), {0.0, 10.0, 100.0});

  // 2. Book Histograms with Axes
  auto h_u = registry.book("h_uniform", u_axis);
  auto h_v = registry.book("h_variable", v_axis);

  auto buffer = registry.create_buffer();

  // 3. Fill using x-values
  registry.fill(buffer, h_u, 5.0, 1.0);   // bin 0 [0,10)
  registry.fill(buffer, h_u, 15.0, 2.0);  // bin 1 [10,20)
  
  registry.fill(buffer, h_v, 5.0, 1.0);   // bin 0 [0,10)
  registry.fill(buffer, h_v, 50.0, 2.0);  // bin 1 [10,100)

  registry.flush(buffer);

  // 4. Verify
  // h_u starts at 0.
  REQUIRE(registry.data().bins()[0].weight() == Approx(1.0));
  REQUIRE(registry.data().bins()[1].weight() == Approx(2.0));

  // h_v starts at 10.
  // h_v bin 0 -> global 10
  // h_v bin 1 -> global 11
  REQUIRE(registry.data().bins()[10].weight() == Approx(1.0));
  REQUIRE(registry.data().bins()[11].weight() == Approx(2.0));
}

TEST_CASE("HistogramRegistry name lookup", "[HistogramRegistry]") {
    HistogramRegistry<> registry;
    auto id = registry.book("my_hist", 10);
    
    REQUIRE_NOTHROW(registry.get_id("my_hist"));
    REQUIRE(registry.get_id("my_hist") == id);
    REQUIRE_THROWS_AS(registry.get_id("non_existent"), std::runtime_error);
}