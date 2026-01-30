#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "kakuhen/histogram/axis.h"

using namespace kakuhen::histogram;
using Catch::Approx;

TEST_CASE("Self-contained Axis", "[axis]") {
  SECTION("Uniform Axis") {
    // 10 bins from 0.0 to 10.0
    UniformAxis<double, uint32_t> axis(10, 0.0, 10.0);

    REQUIRE(axis.n_bins() == 12); // 10 regular + under/overflow
    REQUIRE(axis.index(-1.0) == 0); // Underflow
    REQUIRE(axis.index(0.0) == 1);  // First bin
    REQUIRE(axis.index(0.5) == 1);
    REQUIRE(axis.index(9.9) == 10); // Last bin
    REQUIRE(axis.index(10.0) == 11); // Overflow
    REQUIRE(axis.index(11.0) == 11);

    auto edges = axis.edges();
    REQUIRE(edges.size() == 11);
    REQUIRE(edges.front() == Approx(0.0));
    REQUIRE(edges.back() == Approx(10.0));
    REQUIRE(edges[5] == Approx(5.0));
  }

  SECTION("Variable Axis") {
    // Edges: 0, 2, 5, 10
    VariableAxis<double, uint32_t> axis({0.0, 2.0, 5.0, 10.0});

    REQUIRE(axis.n_bins() == 5); // 3 regular (intervals) + 1 overflow + 1 underflow

    REQUIRE(axis.index(-1.0) == 0); // Underflow
    REQUIRE(axis.index(0.0) == 1);  // [0, 2)
    REQUIRE(axis.index(1.9) == 1);
    REQUIRE(axis.index(2.0) == 2);  // [2, 5)
    REQUIRE(axis.index(4.9) == 2);
    REQUIRE(axis.index(5.0) == 3);  // [5, 10)
    REQUIRE(axis.index(10.0) == 4); // Overflow

    auto edges = axis.edges();
    REQUIRE(edges.size() == 4);
    REQUIRE(edges[0] == 0.0);
    REQUIRE(edges[1] == 2.0);
    REQUIRE(edges[2] == 5.0);
    REQUIRE(edges[3] == 10.0);
  }

  SECTION("Duplicate to external AxisData") {
    VariableAxis<double, uint32_t> axis({0.0, 5.0, 10.0});
    AxisData<double, uint32_t> external_data;

    // Initially external_data is empty
    REQUIRE(external_data.size() == 0);

    // Duplicate
    auto view = axis.duplicate(external_data);

    // external_data should now contain the edges {0.0, 5.0, 10.0}
    REQUIRE(external_data.size() == 3);
    REQUIRE(external_data[0] == 0.0);
    REQUIRE(external_data[1] == 5.0);
    REQUIRE(external_data[2] == 10.0);

    // The view should work correctly with the external data
    REQUIRE(view.n_bins() == 4); // [under, [0,5), [5,10), over]
    REQUIRE(view.index(external_data, -1.0) == 0);
    REQUIRE(view.index(external_data, 2.0) == 1);
    REQUIRE(view.index(external_data, 7.0) == 2);
    REQUIRE(view.index(external_data, 11.0) == 3);
  }
}
