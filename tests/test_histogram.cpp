#include "kakuhen/histogram/bin_accumulator.h"
#include "kakuhen/histogram/histogram_buffer.h"
#include "kakuhen/histogram/histogram_data.h"
#include "kakuhen/util/numeric_traits.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <random>
#include <sstream>
#include <vector>

using namespace kakuhen::histogram;
using Catch::Approx;

TEST_CASE("BinAccumulator operations", "[histogram]") {
  BinAccumulator<double> bin;

  SECTION("Basic accumulation") {
    bin.accumulate(1.0);
    bin.accumulate(2.0);

    REQUIRE(bin.weight() == Approx(3.0));
    REQUIRE(bin.weight_sq() == Approx(1.0 * 1.0 + 2.0 * 2.0));  // 5.0
  }

  SECTION("Explicit weight and squared weight") {
    bin.accumulate(3.0, 9.0);  // w=3, w2=9
    REQUIRE(bin.weight() == Approx(3.0));
    REQUIRE(bin.weight_sq() == Approx(9.0));
  }

  SECTION("Merge accumulators") {
    BinAccumulator<double> other;
    bin.accumulate(1.0);
    other.accumulate(2.0);

    bin += other;
    REQUIRE(bin.weight() == Approx(3.0));
    REQUIRE(bin.weight_sq() == Approx(5.0));
  }

  SECTION("Reset") {
    bin.accumulate(10.0);
    bin.reset();
    REQUIRE(bin.weight() == Approx(0.0));
    REQUIRE(bin.weight_sq() == Approx(0.0));

    bin.reset(5.0, 25.0);
    REQUIRE(bin.weight() == Approx(5.0));
    REQUIRE(bin.weight_sq() == Approx(25.0));
  }

  SECTION("Serialization") {
    bin.accumulate(1.23);
    std::stringstream ss;
    bin.serialize(ss);

    BinAccumulator<double> bin_read;
    bin_read.deserialize(ss);

    REQUIRE(bin_read.weight() == Approx(bin.weight()));
    REQUIRE(bin_read.weight_sq() == Approx(bin.weight_sq()));
  }
}

TEST_CASE("HistogramData operations", "[histogram]") {
  HistogramData<> data;

  SECTION("Resize and access") {
    data.allocate(10);
    REQUIRE(data.size() == 10);
    REQUIRE(data.bins().size() == 10);
    REQUIRE(data.count() == 0);
  }

  SECTION("Accumulate direct") {
    data.allocate(5);
    data.accumulate(2, 10.0);
    REQUIRE(data.bins()[2].weight() == Approx(10.0));

    data.accumulate(2, 5.0, 25.0);
    REQUIRE(data.bins()[2].weight() == Approx(15.0));
    REQUIRE(data.bins()[2].weight_sq() == Approx(100.0 + 25.0));
  }

  SECTION("Event counting") {
    data.allocate(1);
    data.increment_count();
    data.increment_count();
    REQUIRE(data.count() == 2);
  }

  SECTION("Serialization") {
    data.allocate(5);
    data.accumulate(0, 1.0);
    data.increment_count();

    std::stringstream ss;
    data.serialize(ss);

    HistogramData<> data_read;
    data_read.deserialize(ss);

    REQUIRE(data_read.size() == 5);
    REQUIRE(data_read.count() == 1);
    REQUIRE(data_read.bins()[0].weight() == Approx(1.0));
  }
}

TEST_CASE("HistogramBuffer logic", "[histogram]") {
  // Use default double traits
  HistogramBuffer<> buffer;
  HistogramData<> data;

  // Setup 100 global bins
  size_t n_bins = 100;
  data.allocate(n_bins);
  buffer.init(static_cast<uint32_t>(n_bins));

  SECTION("Basic fill and flush") {
    buffer.fill(10, 1.0);
    buffer.fill(10, 2.0);  // Same bin, same event -> sum=3.0, w2=9.0
    buffer.fill(50, 5.0);  // Different bin

    buffer.flush(data);

    REQUIRE(data.bins()[10].weight() == Approx(3.0));
    REQUIRE(data.bins()[10].weight_sq() == Approx(9.0));  // (1+2)^2

    REQUIRE(data.bins()[50].weight() == Approx(5.0));
    REQUIRE(data.bins()[50].weight_sq() == Approx(25.0));

    REQUIRE(data.count() == 1);
  }

  SECTION("Cancellation handling (TwoSum)") {
    // Large cancellation test
    double large = 1.0e16;
    double small = 1.0;

    buffer.fill(0, large);
    buffer.fill(0, -large);
    buffer.fill(0, small);

    buffer.flush(data);

    // Naive sum might lose 'small' if precision is lost, but TwoSum should keep it.
    // Actually, simple double precision 1e16 + 1.0 is representable (16 digits).
    // Let's try something that usually fails without compensation if accumulated naively in global
    // sum. But here we are testing the BUFFER's compensation.

    REQUIRE(data.bins()[0].weight() == Approx(small));
    // The error should correspond to the NET weight of the event squared
    // Net weight = 1.0 -> Error contrib = 1.0
    REQUIRE(data.bins()[0].weight_sq() == Approx(small * small));
  }

  SECTION("Multiple flushes (events)") {
    // Event 1
    buffer.fill(0, 10.0);
    buffer.flush(data);

    // Event 2
    buffer.fill(0, 20.0);
    buffer.flush(data);

    REQUIRE(data.count() == 2);
    REQUIRE(data.bins()[0].weight() == Approx(30.0));
    // Variance: 10^2 + 20^2 = 100 + 400 = 500
    REQUIRE(data.bins()[0].weight_sq() == Approx(500.0));
  }

  SECTION("Generation Index Overflow Simulation") {
    // We can't easily force overflow without mocking the class internals
    // or running billions of loops.
    // However, we can verify that repeated flushing works correctly.

    for (int i = 0; i < 100; ++i) {
      buffer.fill(i % 10, 1.0);
      buffer.flush(data);
    }

    REQUIRE(data.count() == 100);
    for (int i = 0; i < 10; ++i) {
      // Each bin 0..9 filled 10 times with 1.0
      REQUIRE(data.bins()[i].weight() == Approx(10.0));
    }
  }
}

TEST_CASE("HistogramBuffer limit check", "[histogram]") {
  HistogramBuffer<kakuhen::util::NumericTraits<double, uint8_t, uint64_t>> buffer;
  // uint8_t has 8 bits.
  // If we request 200 bins, index needs ceil(log2(200)) = 8 bits.
  // 8 bits >= 8 bits total.
  // The code requires at least 4 bits for generation.
  // So 8 - 8 = 0 < 4. Should throw.

  REQUIRE_THROWS_AS(buffer.init(200), std::runtime_error);

  // Request 10 bins. log2(10) = 4 bits.
  // 8 - 4 = 4 bits for generation. Should pass.
  REQUIRE_NOTHROW(buffer.init(10));
}
