#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "kakuhen/histogram/histogram_registry.h"

using namespace kakuhen::histogram;
using Catch::Approx;

TEST_CASE("HistogramRegistry booking and filling", "[HistogramRegistry]") {
  HistogramRegistry<> registry;

  // 1. Book Histograms
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
  registry.fill(buffer, h1_id, 1, 200.0); // Global idx 1

  // 4. Flush
  registry.flush(buffer);

  // 5. Verify
  REQUIRE(registry.data().bins()[0].weight() == Approx(100.0));
  REQUIRE(registry.data().bins()[1].weight() == Approx(200.0));
  REQUIRE(registry.data().bins()[10].weight() == Approx(0.5));
  REQUIRE(registry.data().bins()[11].weight() == Approx(0.6));
}

TEST_CASE("HistogramRegistry name lookup", "[HistogramRegistry]") {
    HistogramRegistry<> registry;
    auto id = registry.book("my_hist", 10);
    
    REQUIRE_NOTHROW(registry.get_id("my_hist"));
    REQUIRE(registry.get_id("my_hist") == id);
    REQUIRE_THROWS_AS(registry.get_id("non_existent"), std::runtime_error);
}
