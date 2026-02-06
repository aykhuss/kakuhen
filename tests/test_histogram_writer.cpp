#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sstream>
#include <vector>
#include "kakuhen/histogram/axis.h"
#include "kakuhen/histogram/histogram_registry.h"
#include "kakuhen/histogram/histogram_writer.h"

using namespace kakuhen::histogram;
using Catch::Approx;

template <typename NT = kakuhen::util::num_traits_t<>>
class MockWriter : public HistogramWriter<MockWriter<NT>, NT> {
 public:
  using Base = HistogramWriter<MockWriter<NT>, NT>;
  using S = typename Base::size_type;
  using T = typename Base::value_type;
  using U = typename Base::count_type;

  explicit MockWriter(std::ostream& os) : Base(os) {}

  void global_header_impl(const HistogramRegistry<NT>&) {}

  void histogram_header_impl(S, const std::string_view, S, S, S,
                             const std::vector<std::vector<BinRange<T>>>&, U) {}

  void histogram_row_impl(S ibin, const std::vector<BinRange<T>>& bin_range,
                          const std::vector<T>& values, const std::vector<T>& errors) {
    // Basic checks
    if (expected_ndim > 0) {
      REQUIRE(bin_range.size() == expected_ndim);
    }
    
    recorded_bins.push_back({ibin, bin_range});
  }

  void histogram_footer_impl() {}
  void global_footer_impl() {}

  struct RecordedBin {
    S ibin;
    std::vector<BinRange<T>> ranges;
  };

  std::vector<RecordedBin> recorded_bins;
  S expected_ndim = 0;
};

TEST_CASE("HistogramRegistry::write 2D bug reproduction", "[HistogramRegistry][Writer]") {
  HistogramRegistry<> registry;

  // Create Axis objects
  // X axis: 2 bins [0, 10, 20] (+ U/O = 4 bins total)
  UniformAxis<> x_ax(2, 0.0, 20.0); 
  // Y axis: 2 bins [0, 10, 20] (+ U/O = 4 bins total)
  UniformAxis<> y_ax(2, 0.0, 20.0);

  // Book 2D Histogram
  auto h2d = registry.book("h2d", 1, x_ax, y_ax);

  // Total bins = 4 * 4 = 16.
  // X ranges vector size = 4.
  // Y ranges vector size = 4.
  
  // If we iterate flat index from 0 to 15:
  // idx 15 -> (3, 3).
  // If logic is wrong and uses flat index '15' to access ranges[0] (size 4), it will crash or be garbage.

  std::ostringstream oss;
  MockWriter<> writer(oss);
  writer.expected_ndim = 2;

  // This should trigger the out-of-bounds access if the bug exists
  REQUIRE_NOTHROW(registry.write(writer));

  REQUIRE(writer.recorded_bins.size() == 16);
  
  // Verify the last bin (overflow, overflow)
  // idx 15 -> x index 3, y index 3.
  // x range should be > 20 (Overflow)
  // y range should be > 20 (Overflow)
  
  auto last_bin = writer.recorded_bins.back();
  REQUIRE(last_bin.ibin == 15);
  REQUIRE(last_bin.ranges.size() == 2);
  
  // Check that we got the correct ranges (implied check: if we accessed out of bounds, we might get garbage or crash)
  // But let's check specifically for the correct BinKind
  REQUIRE(last_bin.ranges[0].kind == BinKind::Overflow);
  REQUIRE(last_bin.ranges[1].kind == BinKind::Overflow);
}
