#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <array>
#include "kakuhen/histogram/axis.h"
#include "kakuhen/histogram/histogram_registry.h"

using namespace kakuhen::histogram;
using Catch::Approx;

TEST_CASE("HistogramRegistry booking and filling", "[HistogramRegistry]") {
  HistogramRegistry<> registry;

  // 1. Book Histograms (No Axis)
  auto h1_id = registry.book("hist_pt", 1, 10);
  auto h2_id = registry.book("hist_eta", 2, 5); // 5 bins, 2 values each

  REQUIRE(registry.data().size() == 20); // 10 + (5*2) = 20
  REQUIRE(registry.ids().size() == 2);
  REQUIRE(registry.get_name(h1_id) == "hist_pt");

  // 2. Create Buffer
  auto buffer = registry.create_buffer();

  // 3. Fill using local_bin_idx
  std::vector<double> v1 = {100.0};
  registry.fill_by_index(buffer, h1_id, std::span{v1}, static_cast<uint32_t>(0)); // Global idx 0

  std::vector<double> v2 = {0.5, 0.6};
  registry.fill_by_index(buffer, h2_id, std::span{v2}, static_cast<uint32_t>(0)); // Global idx 10, 11 (start of h2)

  // Test scalar fill overload
  registry.fill_by_index(buffer, h1_id, 200.0, static_cast<uint32_t>(1)); // Global idx 1

  // 4. Flush
  registry.flush(buffer);

  // 5. Verify
  REQUIRE(registry.data().get_bin(0).weight() == Approx(100.0));
  REQUIRE(registry.data().get_bin(1).weight() == Approx(200.0));
  REQUIRE(registry.data().get_bin(10).weight() == Approx(0.5));
  REQUIRE(registry.data().get_bin(11).weight() == Approx(0.6));
}

TEST_CASE("HistogramRegistry Axis Integration", "[HistogramRegistry]") {
  HistogramRegistry<> registry;

  // 1. Create self-contained Axis objects
  UniformAxis<> u_ax(10, 0.0, 100.0);
  VariableAxis<> v_ax({0.0, 10.0, 100.0});

  // 2. Book Histograms using Axis objects
  auto h_u = registry.book("h_uniform", 1, u_ax);
  auto h_v = registry.book("h_variable", 1, v_ax);

  auto buffer = registry.create_buffer();

  // 3. Fill using coordinates (new variadic fill: value, coord)
  registry.fill(buffer, h_u, 1.0, 5.0);   // [0,10) -> bin 1
  registry.fill(buffer, h_u, 2.0, 15.0);  // [10,20) -> bin 2

  registry.fill(buffer, h_v, 1.0, 5.0);   // [0,10) -> bin 1
  registry.fill(buffer, h_v, 2.0, 50.0);  // [10,100) -> bin 2

  registry.flush(buffer);

  // 4. Verify
  REQUIRE(registry.get_bin_value(h_u, 1) == Approx(1.0));
  REQUIRE(registry.get_bin_value(h_u, 2) == Approx(2.0));
  REQUIRE(registry.get_bin_value(h_v, 1) == Approx(1.0));
  REQUIRE(registry.get_bin_value(h_v, 2) == Approx(2.0));
}

TEST_CASE("HistogramRegistry Multi-dimensional", "[HistogramRegistry]") {
  HistogramRegistry<> registry;

  // 1. Create Axis objects
  UniformAxis<> x_ax(5, 0.0, 5.0); // 7 bins: [U, 0-1, 1-2, 2-3, 3-4, 4-5, O]
  UniformAxis<> y_ax(2, 0.0, 2.0); // 4 bins: [U, 0-1, 1-2, O]

  // 2. Book 2D Histogram
  auto h2d = registry.book("h2d", 1, x_ax, y_ax);
  auto h2d_mv = registry.book("h2d_mv", 2, x_ax, y_ax);

  REQUIRE(registry.get_ndim(h2d) == 2);
  REQUIRE(registry.get_view(h2d).n_bins() == 28);

  auto buffer = registry.create_buffer();

  // 3. Fill using 2D coordinates (new order: value, x, y)
  // (x=0.5, y=0.5) -> x_bin=1, y_bin=1 -> flat_idx = 1*4 + 1 = 5
  registry.fill(buffer, h2d, 10.0, 0.5, 0.5);

  // Multi-valued 2D
  std::array<double, 2> mv_payload = {1.0, 2.0};
  registry.fill(buffer, h2d_mv, std::span{mv_payload}, 0.5, 0.5);

  registry.flush(buffer);

  // 4. Verify
  REQUIRE(registry.get_bin_value(h2d, 5) == Approx(10.0));

  // h2d_mv: global_idx 5 for bin (1,1), value_idx 0 and 1
  REQUIRE(registry.get_bin_value(h2d_mv, 5, 0) == Approx(1.0));
  REQUIRE(registry.get_bin_value(h2d_mv, 5, 1) == Approx(2.0));
}

TEST_CASE("HistogramRegistry name lookup", "[HistogramRegistry]") {
    HistogramRegistry<> registry;
    auto id = registry.book("my_hist", 1, 10);

    REQUIRE_NOTHROW(registry.get_id("my_hist"));
    REQUIRE(registry.get_id("my_hist") == id);
    REQUIRE_THROWS_AS(registry.get_id("non_existent"), std::runtime_error);
}

TEST_CASE("HistogramRegistry accessors", "[HistogramRegistry]") {
  HistogramRegistry<> registry;
  auto h_id = registry.book("h", 1, 10);
  auto buffer = registry.create_buffer();

  // --- Event 1 ---
  registry.fill_by_index(buffer, h_id, 2.0, static_cast<uint32_t>(2));
  registry.flush(buffer);

  REQUIRE(registry.get_bin_value(h_id, 2) == Approx(2.0));
  REQUIRE(registry.get_bin_error(h_id, 2) == Approx(0.0));
  REQUIRE(registry.get_bin_variance(h_id, 2) == Approx(0.0));

  // --- Event 2 ---
  registry.fill_by_index(buffer, h_id, 6.0, static_cast<uint32_t>(2));
  registry.flush(buffer);

  REQUIRE(registry.get_bin_value(h_id, 2) == Approx(4.0));
  REQUIRE(registry.get_bin_error(h_id, 2) == Approx(2.0));
  REQUIRE(registry.get_bin_variance(h_id, 2) == Approx(4.0));

  const auto& bin = registry.get_bin(h_id, 2);
  REQUIRE(bin.weight() == Approx(8.0));
}
