#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "kakuhen/histogram/histogram_view.h"
#include "kakuhen/histogram/histogram_buffer.h"
#include "kakuhen/histogram/histogram_data.h"
#include <vector>

using namespace kakuhen::histogram;
using Catch::Approx;

TEST_CASE("HistogramView allocation and filling", "[HistogramView]") {
  HistogramData<> data;
  
  // Create views (this should allocate space in data)
  // View 1: 2 bins, 2 values per bin (Total 4)
  HistogramView<> view1(data, 2, 2);
  
  // View 2: 3 bins, 1 value per bin (Total 3)
  HistogramView<> view2(data, 3, 1);

  REQUIRE(data.size() == 7);
  REQUIRE(view1.offset() == 0);
  REQUIRE(view2.offset() == 4);
  REQUIRE(view1.stride() == 2);
  REQUIRE(view2.stride() == 1);

  // Initialize buffer now that we know the total size
  HistogramBuffer<> buffer;
  buffer.init(static_cast<uint32_t>(data.size()));

  SECTION("Filling View 1") {
    // Fill bin 0 of view 1 with {1.0, 2.0}
    // Should map to global 0, 1
    std::vector<double> vals1 = {1.0, 2.0};
    view1.fill(buffer, 0, vals1);

    // Fill bin 1 of view 1 with {3.0, 4.0}
    // Should map to global 2, 3
    std::vector<double> vals2 = {3.0, 4.0};
    view1.fill(buffer, 1, vals2);
    
    buffer.flush(data);

    REQUIRE(data.bins()[0].weight() == Approx(1.0));
    REQUIRE(data.bins()[1].weight() == Approx(2.0));
    REQUIRE(data.bins()[2].weight() == Approx(3.0));
    REQUIRE(data.bins()[3].weight() == Approx(4.0));
  }

  SECTION("Filling View 2") {
    // Fill bin 0 of view 2 (global 4)
    std::vector<double> v1 = {10.0};
    view2.fill(buffer, 0, v1);

    // Fill bin 2 of view 2 (global 6)
    std::vector<double> v2 = {30.0};
    view2.fill(buffer, 2, v2);

    buffer.flush(data);

    REQUIRE(data.bins()[4].weight() == Approx(10.0));
    REQUIRE(data.bins()[5].weight() == Approx(0.0)); // bin 1 skipped
    REQUIRE(data.bins()[6].weight() == Approx(30.0));
  }
  
  SECTION("Interleaved Filling") {
      std::array<double, 2> v1 = {0.1, 0.2};
      view1.fill(buffer, 0, v1);
      std::array<double, 1> v2 = {0.5};
      view2.fill(buffer, 0, v2);
      
      buffer.flush(data);
      
      REQUIRE(data.bins()[0].weight() == Approx(0.1));
      REQUIRE(data.bins()[1].weight() == Approx(0.2));
      REQUIRE(data.bins()[4].weight() == Approx(0.5));
  }

  SECTION("Serialization") {
    HistogramView<> view(10, 5, 2);
    std::stringstream ss;

    SECTION("Without type verification") {
      view.serialize(ss, false);
      
      HistogramView<> restored;
      restored.deserialize(ss, false);

      REQUIRE(restored.offset() == 10);
      REQUIRE(restored.n_bins() == 5);
      REQUIRE(restored.stride() == 2);
    }

    SECTION("With type verification") {
      view.serialize(ss, true);
      
      HistogramView<> restored;
      restored.deserialize(ss, true);

      REQUIRE(restored.offset() == 10);
      REQUIRE(restored.n_bins() == 5);
      REQUIRE(restored.stride() == 2);
    }

    SECTION("Type mismatch detection") {
      view.serialize(ss, true);
      
      // Use different traits (e.g. float coordinate) to trigger mismatch
      using FloatTraits = kakuhen::util::NumericTraits<float, uint32_t, uint64_t>;
      HistogramView<FloatTraits> wrong_view;
      
      REQUIRE_THROWS_AS(wrong_view.deserialize(ss, true), std::runtime_error);
    }
  }
}
