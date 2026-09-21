#include "kakuhen/integrator/basin.h"
#include "kakuhen/integrator/vegas.h"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace kakuhen::integrator;

// Exposes the grid and writes data streams from hand-set accumulators.
// `write_data_stream` mirrors the `.khd` field order of Vegas and Basin.
template <typename Integrator>
struct ValidationFixture : Integrator {
  using Integrator::grid_;
  using Integrator::Integrator;
  using S = typename Integrator::size_type;
  using U = typename Integrator::count_type;
  static constexpr bool is_vegas = requires(const Integrator& i) { i.ndiv(); };

  kakuhen::ndarray::NDArray<GridAccumulator<double, U>, S> cells{grid_.shape()};
  IntegralAccumulator<double, U> sums;
  U count = 0;

  /// the last grid dimension, whose change leaves the grid itself untouched
  S& last_ndiv() {
    if constexpr (is_vegas) {
      return this->ndiv_;
    } else {
      return this->ndiv2_;
    }
  }

  /// consistent grid data of `n` samples: the first cell of every row holds all of them
  void set_samples(U n, double value) {
    S row_size;
    if constexpr (is_vegas) {
      row_size = this->ndiv();
    } else {
      row_size = this->ndiv1() * this->ndiv2();
    }
    count = n;
    for (S i = 0; i < cells.size(); ++i) {
      cells[i].reset(i % row_size == 0 ? value : 0.0, i % row_size == 0 ? n : 0);
    }
  }

  void write_data_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    serialize_one<S>(out, this->ndim());
    if constexpr (is_vegas) {
      serialize_one<S>(out, this->ndiv());
    } else {
      serialize_one<S>(out, this->ndiv1());
      serialize_one<S>(out, this->ndiv2());
    }
    serialize_one<kakuhen::util::HashValue_t>(out, this->hash().value());
    sums.serialize(out);
    serialize_one<U>(out, count);
    cells.serialize(out);
  }
};

/// the full grid state and accumulated data as bytes
static std::string snapshot(const auto& integrator) {
  std::stringstream out;
  integrator.write_state_stream(out);
  integrator.write_data_stream(out);
  return out.str();
}

TEMPLATE_TEST_CASE("Failed state reads leave integrators unchanged", "[validation]", Vegas<>,
                   Basin<>) {
  ValidationFixture<TestType> source(2);
  TestType target(3);
  target.set_seed(42);
  target.integrate([](const Point<>&) { return 2.0; },
                   {.neval = 10, .niter = 1, .adapt = false, .verbosity = 0});
  const auto before = snapshot(target);
  const auto require_rejected = [&] {
    std::stringstream in;
    source.write_state_stream(in);
    REQUIRE_THROWS_AS(target.read_state_stream(in), std::runtime_error);
    REQUIRE(snapshot(target) == before);
  };
  SECTION("truncated state") {
    std::stringstream ss;
    source.write_state_stream(ss);
    const std::string full = ss.str();
    for (const std::size_t len : {std::size_t(3), full.size() / 2, full.size() - 1}) {
      std::stringstream truncated(full.substr(0, len));
      REQUIRE_THROWS_AS(target.read_state_stream(truncated), std::runtime_error);
      REQUIRE(snapshot(target) == before);
    }
  }
  SECTION("invalid grid dimensions") {
    source.last_ndiv() = 1;
    require_rejected();
  }
  SECTION("grid shape mismatch") {
    source.last_ndiv() -= 1;
    require_rejected();
  }
  SECTION("non-finite or out-of-range boundary") {
    for (const double value : {-0.1, 1.1, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
      source.grid_[0] = value;
      require_rejected();
    }
  }
  SECTION("decreasing boundaries") {
    source.grid_[0] = 0.9;
    require_rejected();
  }
  SECTION("wrong terminal boundary") {
    source.grid_[source.grid_.size() - 1] = 0.999;
    require_rejected();
  }
}

TEST_CASE("Basin rejects invalid sampling forests before mutation", "[basin][validation]") {
  struct Fixture : Basin<> {
    Fixture() : Basin<>(3, 2, 2) {}
    using Basin<>::order_;
  } source;
  Basin<> target(2);
  const auto before = snapshot(target);
  SECTION("out-of-range dimension") {
    source.order_(1, 0) = 3;
  }
  SECTION("duplicate roots and missing dimension") {
    source.order_(1, 0) = 0;
    source.order_(1, 1) = 0;
  }
  SECTION("rootless cycle") {
    source.order_(0, 0) = 2;
    source.order_(1, 0) = 0;
    source.order_(2, 0) = 1;
  }
  SECTION("cycle disconnected from a root") {
    source.order_(1, 0) = 2;
    source.order_(2, 0) = 1;
  }
  std::stringstream in;
  source.write_state_stream(in);
  REQUIRE_THROWS_AS(target.read_state_stream(in), std::runtime_error);
  REQUIRE(snapshot(target) == before);
}

TEMPLATE_TEST_CASE("Failed data merges leave integrators unchanged", "[validation]", Vegas<>,
                   Basin<>) {
  using U = typename TestType::count_type;
  constexpr U max_count = std::numeric_limits<U>::max();
  ValidationFixture<TestType> source(2), initial(2);
  TestType target(2);
  source.sums.reset(1, 1, 1);
  source.set_samples(1, 1.0);
  initial.set_samples(1, 1.0);

  SECTION("corrupt or incompatible data") {
    std::stringstream initial_data;
    initial.write_data_stream(initial_data);
    target.accumulate_data_stream(initial_data);
    const auto before = snapshot(target);
    const auto require_rejected = [&] {
      std::stringstream in;
      source.write_data_stream(in);
      REQUIRE_THROWS_AS(target.accumulate_data_stream(in), std::runtime_error);
      REQUIRE(snapshot(target) == before);
    };
    SECTION("truncated data") {
      std::stringstream ss;
      source.write_data_stream(ss);
      std::stringstream truncated(ss.str().substr(0, ss.str().size() - 1));
      REQUIRE_THROWS_AS(target.accumulate_data_stream(truncated), std::runtime_error);
      REQUIRE(snapshot(target) == before);
    }
    SECTION("invalid incoming cell") {
      for (const double value : {-1.0, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
        source.cells[0].reset(value, 1);
        require_rejected();
      }
    }
    SECTION("negative integral sum of squares") {
      source.sums.reset(2, -2, 2);
      require_rejected();
    }
    SECTION("non-finite integral sums") {
      for (const double value :
           {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        source.sums.reset(value, 1, 1);
        require_rejected();
        source.sums.reset(1, value, 1);
        require_rejected();
      }
    }
    SECTION("nonzero integral sums without samples") {
      source.sums.reset(1, 0, 0);
      require_rejected();
      source.sums.reset(0, 1, 0);
      require_rejected();
    }
    SECTION("positive cell value without samples") {
      source.cells[1].reset(100, 0);
      require_rejected();
    }
    SECTION("positive cell value with zero adaptation count") {
      source.set_samples(0, 0);
      source.cells[0].reset(100, 0);
      require_rejected();
    }
    SECTION("cell counts inconsistent with the adaptation count") {
      source.cells[0].reset(1.0, 2);
      require_rejected();
    }
    SECTION("cell counts that wrap around to the adaptation count") {
      source.cells[1].reset(0.0, max_count);
      require_rejected();
    }
    SECTION("grid of a different shape") {
      ValidationFixture<TestType> other(3);
      other.set_samples(1, 1.0);
      std::stringstream in;
      other.write_data_stream(in);
      REQUIRE_THROWS_AS(target.accumulate_data_stream(in), std::runtime_error);
      REQUIRE(snapshot(target) == before);
    }
  }

  SECTION("overflowing sums or counts") {
    SECTION("cell sum") {
      source.set_samples(1, 1e308);
      initial.set_samples(1, 1e308);
    }
    SECTION("adaptation count") {
      initial.set_samples(max_count, 0.0);
    }
    SECTION("integral sum") {
      source.sums.reset(0, 1e308, 1);
      initial.sums.reset(0, 1e308, 1);
    }
    SECTION("integral count") {
      initial.sums.reset(0, 0, max_count);
    }
    std::stringstream initial_data;
    initial.write_data_stream(initial_data);
    target.accumulate_data_stream(initial_data);
    const auto before = snapshot(target);
    std::stringstream in;
    source.write_data_stream(in);
    REQUIRE_THROWS_AS(target.accumulate_data_stream(in), std::overflow_error);
    REQUIRE(snapshot(target) == before);
  }
}

TEST_CASE("Vegas keeps a degenerate dimension and adapts subsequent dimensions",
          "[vegas][validation]") {
  ValidationFixture<Vegas<>> source(2, 4), veg(2, 4);
  veg.set_options({.verbosity = 0});
  veg.set_alpha(2000.0);
  source.count = 4;
  for (std::size_t i = 0; i < 4; ++i) {
    source.cells(0, i).reset(1.0, 1);
    source.cells(1, i).reset(i == 0 ? 100.0 : 1.0, 1);
  }
  std::stringstream data;
  source.write_data_stream(data);
  veg.accumulate_data_stream(data);
  const std::vector<double> before(veg.grid_.begin(), veg.grid_.end());
  veg.adapt();
  for (std::size_t i = 0; i < 4; ++i) {
    REQUIRE(veg.grid_(0, i) == before[i]);
  }
  REQUIRE(veg.grid_(1, 0) != before[4]);
  REQUIRE(veg.grid_(1, 3) == 1.0);
}

TEMPLATE_TEST_CASE("State loading permits rounded repeated boundaries", "[validation]", Vegas<>,
                   Basin<>) {
  ValidationFixture<TestType> source(2);
  source.grid_[0] = source.grid_[1];
  TestType target(3);
  std::stringstream in;
  source.write_state_stream(in);
  REQUIRE_NOTHROW(target.read_state_stream(in));
  REQUIRE(target.ndim() == source.ndim());
  REQUIRE(target.hash().value() == source.hash().value());
}

TEST_CASE("Basin loads an unordered forest and rebuilds sampling views", "[basin][validation]") {
  struct Fixture : Basin<> {
    Fixture() : Basin<>(3, 2, 2) {}
    using Basin<>::order_;
  } source;
  source.order_(0, 0) = 2;
  source.order_(0, 1) = 2;
  source.order_(1, 0) = 2;
  source.order_(1, 1) = 0;
  source.order_(2, 0) = 0;
  source.order_(2, 1) = 1;
  std::stringstream canonical;
  source.write_state_stream(canonical);
  for (std::size_t column = 0; column < 2; ++column) {
    std::swap(source.order_(0, column), source.order_(2, column));
  }
  std::stringstream unordered;
  source.write_state_stream(unordered);
  Basin<> target(2);
  target.read_state_stream(unordered);
  std::stringstream loaded;
  target.write_state_stream(loaded);
  REQUIRE(loaded.str() == canonical.str());
  target.set_options({.verbosity = 0});
  const auto result = target.integrate([](const Point<>&) { return 1.0; },
                                       {.neval = 10, .niter = 1, .adapt = false});
  REQUIRE(result.value() == 1.0);
}
