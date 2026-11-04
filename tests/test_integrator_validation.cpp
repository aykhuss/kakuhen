#include "kakuhen/integrator/basin.h"
#include "kakuhen/integrator/vegas.h"
#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <random>
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
  using Integrator::uniform_distribution_;
  using S = typename Integrator::size_type;
  using U = typename Integrator::count_type;
  static constexpr bool is_vegas = requires(const Integrator& i) { i.ndiv(); };

  kakuhen::ndarray::NDArray<typename Integrator::grid_acc_type, S> cells{grid_.shape()};
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
      serialize_one<kakuhen::util::HashValue_t>(out, this->hash().value());
    } else {
      serialize_one<S>(out, this->ndiv1());
      serialize_one<S>(out, this->ndiv2());
      serialize_one<kakuhen::util::HashValue_t>(out, this->map_hash());
    }
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

template <typename Integrator>
struct CollectionFixture : Integrator {
  using Integrator::Integrator;
  using Integrator::result_;
};

TEMPLATE_TEST_CASE("Pending integral statistics include every collected batch", "[validation]",
                   Vegas<>, Basin<>) {
  CollectionFixture<TestType> source(1);
  const auto first = source.integrate([](const Point<>& p) { return double(p.sample_index + 1); },
                                      {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});
  REQUIRE(first.count() == 2);
  REQUIRE(first.value() == 1.5);

  unsigned copies = 1;
  bool interrupted = false;
  SECTION("separate calls") {}
  SECTION("read data before continuing") {
    std::stringstream data;
    source.write_data_stream(data);
    source.clear_data();
    source.read_data_stream(data);
  }
  SECTION("merge data before continuing") {
    std::stringstream data;
    source.write_data_stream(data);
    source.accumulate_data_stream(data);
    copies = 2;
  }
  SECTION("an exception retains the completed prefix") {
    interrupted = true;
    REQUIRE_THROWS_AS(source.integrate(
                          [](const Point<>& p) {
                            if (p.sample_index == 2) throw std::runtime_error("stop");
                            return 3.0;
                          },
                          {.neval = 3, .niter = 1, .adapt = false, .verbosity = 0}),
                      std::runtime_error);
    REQUIRE(source.result_.count() == 4);
    REQUIRE(source.result_.f_.result() == 9.0);
    REQUIRE(source.result_.f2_.result() == 23.0);
  }
  const auto last = source.integrate(
      [interrupted](const Point<>&) { return interrupted ? 4.0 : 3.0; },
      {.neval = interrupted ? 1u : 3u, .niter = 1, .adapt = false, .verbosity = 0});
  REQUIRE(last.count() == (interrupted ? 1 : 3));
  REQUIRE(last.value() == (interrupted ? 4.0 : 3.0));
  REQUIRE(source.result_.count() == 2 * copies + 3);
  REQUIRE(source.result_.f_.result() == 3 * copies + (interrupted ? 10 : 9));
  REQUIRE(source.result_.f2_.result() == 5 * copies + (interrupted ? 34 : 27));
}

TEMPLATE_TEST_CASE("Progress checkpoints include the current collection prefix",
                   "[validation][progress]", Vegas<>, Basin<>) {
  CollectionFixture<TestType> source(1);
  source.integrate([](const Point<>&) { return 2.0; },
                   {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});
  std::stringstream pending;
  source.write_data_stream(pending);
  bool stop = false, throwing = false, frozen = false, non_finite = false;
  SECTION("continue through multiple iterations") {}
  SECTION("stop at a milestone") {
    stop = true;
  }
  SECTION("throw at a milestone") {
    throwing = true;
  }
  SECTION("frozen calls leave pending data unchanged") {
    frozen = true;
  }
  SECTION("non-finite frozen calls leave pending data unchanged") {
    frozen = true;
    non_finite = true;
  }
  std::vector<std::string> checkpoints;
  const auto result = source.integrate(
      [non_finite](const Point<>& p) {
        return non_finite ? std::numeric_limits<double>::quiet_NaN() : double(p.sample_index + 3);
      },
      {.neval = 4,
       .niter = 2,
       .adapt = false,
       .frozen = frozen,
       .verbosity = 0,
       .progress_step = 0.5},
      [&](const typename TestType::progress_event_type& event) {
        if (event.kind != ProgressEventKind::EVAL_MILESTONE) return EventSignal::NONE;
        std::stringstream out;
        source.write_data_stream(out);
        checkpoints.push_back(out.str());
        if (throwing) throw std::runtime_error("stop");
        return stop ? EventSignal::STOP : EventSignal::NONE;
      });
  REQUIRE(checkpoints.size() == (stop || throwing ? 1 : 4));
  REQUIRE(result.count() == 2 * checkpoints.size());
  if (non_finite) {
    REQUIRE(std::isnan(result.value()));
  } else {
    REQUIRE(result.value() == (stop || throwing ? 3.5 : 4.5));
  }
  constexpr double sums[] = {11, 22, 29, 40};
  constexpr double squares[] = {33, 94, 119, 180};
  for (std::size_t i = 0; i < checkpoints.size(); ++i) {
    CAPTURE(i, frozen);
    if (frozen) {
      REQUIRE(checkpoints[i] == pending.str());
      continue;
    }
    CollectionFixture<TestType> loaded(1);
    std::stringstream in(checkpoints[i]);
    loaded.read_data_stream(in);
    REQUIRE(loaded.result_.count() == 4 + 2 * i);
    REQUIRE(loaded.result_.f_.result() == sums[i]);
    REQUIRE(loaded.result_.f2_.result() == squares[i]);
  }
  std::stringstream after;
  source.write_data_stream(after);
  REQUIRE(after.str() == checkpoints.back());
}

TEMPLATE_TEST_CASE("Integrand checkpoints contain the previously completed samples", "[validation]",
                   Vegas<>, Basin<>) {
  CollectionFixture<TestType> source(1), loaded(1);
  std::stringstream checkpoint;
  const auto result = source.integrate(
      [&](const Point<>& p) {
        if (p.sample_index == 2) source.write_data_stream(checkpoint);
        return double(p.sample_index + 1);
      },
      {.neval = 4, .niter = 1, .adapt = false, .verbosity = 0});
  REQUIRE(result.count() == 4);
  REQUIRE(result.value() == 2.5);
  loaded.read_data_stream(checkpoint);
  REQUIRE(loaded.result_.count() == 2);
  REQUIRE(loaded.result_.f_.result() == 3.0);
  REQUIRE(loaded.result_.f2_.result() == 5.0);
}

TEMPLATE_TEST_CASE("Progress callbacks can clear pending data during collection",
                   "[validation][progress]", Vegas<>, Basin<>) {
  CollectionFixture<TestType> source(1), loaded(1);
  std::stringstream empty;
  source.write_data_stream(empty);
  bool stop = false, throwing = false;
  SECTION("continue collecting after clearing") {}
  SECTION("stop immediately after clearing") {
    stop = true;
  }
  SECTION("throw immediately after clearing") {
    throwing = true;
  }
  bool cleared = false;
  std::stringstream checkpoint;
  const auto result = source.integrate(
      [](const Point<>& p) { return double(p.sample_index + 1); },
      {.neval = 4, .niter = 1, .adapt = false, .verbosity = 0, .progress_step = 0.5},
      [&](const typename TestType::progress_event_type& event) {
        if (event.kind != ProgressEventKind::EVAL_MILESTONE || cleared) return EventSignal::NONE;
        source.clear_data();
        source.write_data_stream(checkpoint);
        cleared = true;
        if (throwing) throw std::runtime_error("stop");
        return stop ? EventSignal::STOP : EventSignal::NONE;
      });
  REQUIRE(cleared);
  REQUIRE(checkpoint.str() == empty.str());
  REQUIRE(result.count() == (stop || throwing ? 2 : 4));
  REQUIRE(result.value() == (stop || throwing ? 1.5 : 2.5));
  REQUIRE(result.at(0).f2_.result() == (stop || throwing ? 5.0 : 30.0));

  std::stringstream pending;
  source.write_data_stream(pending);
  REQUIRE_NOTHROW(loaded.read_data_stream(pending));
  REQUIRE(loaded.result_.count() == (stop || throwing ? 0 : 2));
  REQUIRE(loaded.result_.f_.result() == (stop || throwing ? 0.0 : 7.0));
  REQUIRE(loaded.result_.f2_.result() == (stop || throwing ? 0.0 : 25.0));
  if (stop || throwing) REQUIRE(pending.str() == empty.str());
}

TEMPLATE_TEST_CASE("Non-finite pending batches remain invalid after later collection",
                   "[validation][progress]", Vegas<>, Basin<>) {
  TestType source(1), target(1);
  bool interrupted = false;
  SECTION("completed batch") {}
  SECTION("interrupted batch") {
    interrupted = true;
  }
  std::vector<std::string> checkpoints;
  const auto save = [&] {
    std::stringstream out;
    source.write_data_stream(out);
    checkpoints.push_back(out.str());
  };
  const auto collect = [&] {
    return source.integrate(
        [interrupted](const Point<>& p) {
          if (interrupted && p.sample_index == 2) throw std::runtime_error("stop");
          return p.sample_index == 0 ? std::numeric_limits<double>::quiet_NaN() : 1.0;
        },
        {.neval = 4, .niter = 1, .adapt = false, .verbosity = 0, .progress_step = 0.5},
        [&](const typename TestType::progress_event_type& event) {
          if (event.kind == ProgressEventKind::EVAL_MILESTONE) save();
          return EventSignal::NONE;
        });
  };
  if (interrupted) {
    REQUIRE_THROWS_AS(collect(), std::runtime_error);
  } else {
    REQUIRE(std::isnan(collect().value()));
  }
  const auto finite = source.integrate([](const Point<>&) { return 3.0; },
                                       {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});
  REQUIRE(finite.count() == 2);
  REQUIRE(finite.value() == 3.0);
  save();
  REQUIRE(checkpoints.size() == (interrupted ? 2 : 3));
  const auto before = snapshot(target);
  for (const auto& bytes : checkpoints) {
    std::stringstream in(bytes);
    REQUIRE_THROWS_AS(target.accumulate_data_stream(in), std::runtime_error);
    REQUIRE(snapshot(target) == before);
  }
  source.clear_data();
  source.integrate([](const Point<>&) { return 1.0; },
                   {.neval = 1, .niter = 1, .adapt = false, .verbosity = 0});
  std::stringstream valid;
  source.write_data_stream(valid);
  REQUIRE_NOTHROW(target.read_data_stream(valid));
}

TEST_CASE("Basin collection uses the first containing fine bin", "[basin][validation]") {
  for (unsigned fine : {16u, 32u, 33u, 128u}) {  // both sides of `Basin::MAX_SCAN`
    for (double x : {0.0, 0.5}) {
      CAPTURE(fine, x);
      ValidationFixture<Basin<>> actual(2, 2, fine), expected(2, 2, fine);
      actual.uniform_distribution_ = std::uniform_real_distribution<double>(x, x);
      actual.integrate([](const Point<>& p) { return 1.0 / p.weight; },
                       {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});
      expected.sums.reset(2.0, 2.0, 2);
      expected.count = 2;
      const unsigned row = x == 0.0 ? 0 : 1;
      const unsigned bin = x == 0.0 ? 0 : (fine - 1) / 2;
      // f^2 = 1 is stored as G = f^2 / w, with w the weight of the diagonal cell of both dims
      const double low = row > 0 ? expected.grid_(0, 0, 0, fine - 1) : 0.0;
      const double g = 1.0 / (2.0 * fine * (expected.grid_(0, 0, row, 0) - low));
      for (unsigned d = 0; d < 2; ++d) {
        expected.cells(d, d, row, 0).reset(2 * g, 2);
        expected.cells(d, 1 - d, row, bin).reset(2 * g, 2);
      }
      std::stringstream observed, reference;
      actual.Basin<>::write_data_stream(observed);
      expected.write_data_stream(reference);
      REQUIRE(observed.str() == reference.str());
    }
  }
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

  SECTION("overflowing sums") {
    SECTION("cell sum") {
      source.set_samples(1, 1e308);
      initial.set_samples(1, 1e308);
    }
    SECTION("integral sum") {
      source.sums.reset(0, 1e308, 1);
      initial.sums.reset(0, 1e308, 1);
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

TEMPLATE_TEST_CASE("Data imports reject count overflow", "[validation]", Vegas<>, Basin<>) {
  using U = typename TestType::count_type;
  constexpr U MAX_COUNT = std::numeric_limits<U>::max();
  ValidationFixture<TestType> initial(2), source(2);
  TestType target(2);
  SECTION("integral count") {
    initial.sums.reset(0, 0, MAX_COUNT - 1);
    source.sums.reset(0, 0, 1);
  }
  SECTION("adaptation count") {
    initial.set_samples(MAX_COUNT - 1, 0);
    source.set_samples(1, 0);
  }
  std::stringstream initial_data;
  initial.write_data_stream(initial_data);
  target.accumulate_data_stream(initial_data);

  // Reaching the maximum is valid; importing one more sample must leave it intact.
  std::stringstream incoming;
  source.write_data_stream(incoming);
  REQUIRE_NOTHROW(target.accumulate_data_stream(incoming));
  const auto before = snapshot(target);
  incoming.clear();
  incoming.seekg(0);
  REQUIRE_THROWS_AS(target.accumulate_data_stream(incoming), std::overflow_error);
  REQUIRE(snapshot(target) == before);
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

TEMPLATE_TEST_CASE("Non-finite samples reach the result but not the adaptive grid", "[validation]",
                   Vegas<>, Basin<>) {
  const auto with_nan = [](const Point<>& p) {
    return p.sample_index % 7 == 3 ? std::numeric_limits<double>::quiet_NaN() : 1.0 + p.x[0];
  };
  ValidationFixture<TestType> adapting(2);
  adapting.set_options({.verbosity = 0});
  const auto result = adapting.integrate(with_nan, {.neval = 200, .niter = 2, .adapt = true});
  REQUIRE(std::isnan(result.value()));
  REQUIRE(std::ranges::all_of(adapting.grid_, [](double x) { return std::isfinite(x); }));

  // the data file carries the NaN integral as well, so merging it fails loudly
  TestType source(2), target(2);
  source.set_options({.verbosity = 0});
  source.integrate(with_nan, {.neval = 200, .niter = 1, .adapt = false});
  std::stringstream data;
  source.write_data_stream(data);
  REQUIRE_THROWS_AS(target.accumulate_data_stream(data), std::runtime_error);
}
