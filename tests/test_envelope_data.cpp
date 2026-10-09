#include "kakuhen/integrator/basin_generator.h"
#include "kakuhen/integrator/vegas_generator.h"
#include "test_support.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

using namespace kakuhen::integrator;
using Catch::Approx;
using test_support::read_file;
using test_support::TempDir;
using test_support::write_file;

namespace {
template <typename G>
struct Fixture : G {
  using G::envelope_;
  using G::G;
  using G::grid_;

  void configure() {
    this->set_options({.frozen = true, .verbosity = 0});
    if constexpr (std::same_as<G, BasinGenerator<>>) {
      // exercise conditional rows and a chain, not just independent roots
      this->nblocks_ = 1;
      for (typename G::size_type i = 1; i < this->ndim_; ++i)
        this->order_(i, 0) = i - 1;
      for (typename G::size_type i = 0; i < this->ndim_; ++i)
        std::copy_n(&this->grid_(this->order_(i, 0), this->order_(i, 1), 0, 0), this->ndiv0_,
                    &this->ordered_grid_(i, 0, 0));
    } else {
      // non-unit Jacobians make sure records store |J f| and not |f|
      for (typename G::size_type i = 0; i < this->ndim_; ++i)
        this->grid_(i, 0) = 0.05;
    }
    this->initialize_envelope(1.0);
    this->set_seed(73);
  }
};

template <typename G>
auto make_generator() {
  auto gen = [] {
    if constexpr (std::same_as<G, BasinGenerator<>>)
      return Fixture<G>(3, 2, 2);
    else
      return Fixture<G>(3, 4);
  }();
  gen.configure();
  return gen;
}

std::vector<double> factors(const auto& gen) {
  return {gen.envelope_.begin(), gen.envelope_.end()};
}

std::string snapshot(const auto& gen) {
  std::ostringstream out;
  gen.write_state_stream(out);
  gen.write_rng_state_stream(out);
  return out.str();
}

double integrand(const Point<>& p) {
  return p.sample_index % 5 == 0 ? 0.0 : -(2.0 + 4.0 * p.x[0] + p.x[1] * p.x[2]);
}

}  // namespace

TEMPLATE_TEST_CASE("Deferred envelope replay matches immediate training for the same observations",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  auto immediate = make_generator<TestType>();
  auto deferred = make_generator<TestType>();
  const auto baseline = factors(deferred);
  const auto online = immediate.raise_envelope(integrand, 512);
  const auto collected = deferred.collect_envelope(integrand, 512);
  REQUIRE(factors(deferred) == baseline);
  REQUIRE(deferred.envelope_ready());
  REQUIRE(collected.n_evaluations() == 512);
  REQUIRE(collected.n_violations() >= online.n_violations());
  REQUIRE(collected.n_violations() < 512);  // omitted zero observations
  REQUIRE(collected.n_raised() == 0);
  REQUIRE(online.n_raised() == online.n_violations());
  const auto replay = deferred.adapt_envelope();
  REQUIRE(replay.n_evaluations() == 512);
  // the replay reports the batch's violation rate, but raises like online training
  REQUIRE(replay.n_violations() == collected.n_violations());
  REQUIRE(replay.n_raised() == online.n_raised());
  REQUIRE(factors(deferred) == factors(immediate));
  REQUIRE(deferred.envelope_volume() == immediate.envelope_volume());
  REQUIRE(deferred.abs_integral_estimate().count() == 512);
  REQUIRE(deferred.abs_integral_estimate().value() == immediate.abs_integral_estimate().value());
  REQUIRE_FALSE(deferred.has_envelope_data());
  REQUIRE(deferred.n_envelope_records() == 0);
  const auto after = snapshot(deferred);
  REQUIRE(deferred.adapt_envelope().n_evaluations() == 0);
  REQUIRE(snapshot(deferred) == after);
}

TEMPLATE_TEST_CASE("Envelope record limits stop at the last retained observation across calls",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  auto gen = make_generator<TestType>();
  auto alternating = [](const Point<>& p) { return (p.sample_index % 2 ? 3.0 : 0.5) / p.weight; };
  auto result = gen.collect_envelope(alternating, 800, 3);
  REQUIRE(result.status() == EnvelopeStatus::RECORD_LIMIT_REACHED);
  REQUIRE(result.n_evaluations() == 6);
  REQUIRE(result.n_violations() == 3);
  REQUIRE(gen.n_envelope_records() == 3);
  REQUIRE(gen.abs_integral_estimate().count() == 6);
  REQUIRE(gen.abs_integral_estimate().value() == Approx(1.75));
  const auto before = snapshot(gen);
  result = gen.collect_envelope(alternating, 800, 3);
  REQUIRE(result.n_evaluations() == 0);
  REQUIRE(result.status() == EnvelopeStatus::RECORD_LIMIT_REACHED);
  REQUIRE(snapshot(gen) == before);
  result = gen.collect_envelope(alternating, 100, 5);
  REQUIRE(result.n_violations() == 2);
  REQUIRE(result.n_evaluations() == 4);
  REQUIRE(gen.n_envelope_records() == 5);
  gen.clear_envelope_data();
  REQUIRE(gen.abs_integral_estimate().count() == 10);
  REQUIRE(gen.collect_envelope(alternating, 10, 0).n_evaluations() == 0);
  REQUIRE_FALSE(gen.has_envelope_data());
}

TEMPLATE_TEST_CASE("Collection exhausts its budget and preserves zero-violation batches",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-env-data");
  auto gen = make_generator<TestType>();
  gen.initialize_envelope([](const Point<>& p) { return 1.0 / p.weight; }, 7);
  const auto result = gen.collect_envelope([](const Point<>& p) { return 0.5 / p.weight; }, 12);
  REQUIRE(result.status() == EnvelopeStatus::BUDGET_EXHAUSTED);
  REQUIRE(result.n_evaluations() == 12);
  REQUIRE(result.n_violations() == 0);
  REQUIRE(gen.has_envelope_data());
  REQUIRE(gen.n_envelope_records() == 0);
  REQUIRE_THROWS(gen.initialize_envelope(1.0));
  REQUIRE_THROWS(gen.scale_envelope(0.5));
  gen.raise_envelope([](const Point<>& p) { return 2.0 / p.weight; }, 4);
  gen.save_envelope(files.dir / "batch.khe");
  REQUIRE(gen.has_envelope_data());
  auto receiver = make_generator<TestType>();
  receiver.merge_envelope(files.dir / "batch.khe");
  REQUIRE(receiver.abs_integral_estimate().count() == 12);  // excludes seed and online history
  REQUIRE(receiver.abs_integral_estimate().value() == Approx(0.5));
  REQUIRE_FALSE(receiver.has_envelope_data());
  const auto count = gen.abs_integral_estimate().count();
  gen.clear_envelope_data();
  REQUIRE(gen.abs_integral_estimate().count() == count);
  REQUIRE_NOTHROW(gen.scale_envelope(0.5));
  REQUIRE_NOTHROW(gen.initialize_envelope(1.0));
  const auto online = gen.optimize_envelope([](const Point<>&) { return 0.0; }, 4, 3);
  REQUIRE(online.status() == EnvelopeStatus::TARGET_REACHED);
  REQUIRE(online.n_evaluations() == 4);
}

TEMPLATE_TEST_CASE("Maximum merging replays only imported data and counts the batch once",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-env-data");
  auto source = make_generator<TestType>();
  auto receiver = make_generator<TestType>();
  auto reference = make_generator<TestType>();
  source.scale_envelope(2.0);  // different baselines are fine
  source.save_envelope(files.dir / "baseline.khe");
  source.collect_envelope(integrand, 128);
  source.save_envelope(files.dir / "batch.khe");
  auto local = [](const Point<>& p) { return 100.0 / p.weight; };
  receiver.collect_envelope(local, 3);
  reference.collect_envelope(local, 3);
  reference.merge_envelope(files.dir / "baseline.khe");
  reference.set_seed(73);
  reference.raise_envelope(integrand, 128);
  receiver.merge_envelope(files.dir / "batch.khe");
  REQUIRE(factors(receiver) == factors(reference));
  REQUIRE(receiver.n_envelope_records() == 3);
  REQUIRE(receiver.abs_integral_estimate().count() == 131);
  REQUIRE(receiver.abs_integral_estimate().value() ==
          Approx(reference.abs_integral_estimate().value()));
  // an imported batch is consumed, not queued or exported again
  receiver.save_envelope(files.dir / "local-only.khe");
  auto third = make_generator<TestType>();
  third.merge_envelope(files.dir / "local-only.khe");
  REQUIRE(third.abs_integral_estimate().count() == 3);
  REQUIRE(third.abs_integral_estimate().value() == Approx(100.0));
  REQUIRE(source.n_envelope_records() > 0);          // save was non-consuming
  receiver.merge_envelope(files.dir / "batch.khe");  // duplicates are the caller's responsibility
  REQUIRE(receiver.abs_integral_estimate().count() == 259);
  const auto stats = receiver.abs_integral_estimate().count();
  receiver.adapt_envelope();
  REQUIRE(receiver.abs_integral_estimate().count() == stats);
  REQUIRE_FALSE(receiver.has_envelope_data());
}

TEMPLATE_TEST_CASE("Checkpoints retain pending batches and leave trailing user data readable",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-env-data");
  auto original = make_generator<TestType>();
  original.collect_envelope(integrand, 64);
  original.scale_envelope(1.25);  // monotonic changes preserve the batch
  original.save(files.dir / "state.khs");
  kakuhen::util::write_user_data(files.dir / "state.khs", 42, "extra");
  auto restored = make_generator<TestType>();
  restored.load(files.dir / "state.khs");
  REQUIRE(restored.has_envelope_data());
  REQUIRE(restored.n_envelope_records() == original.n_envelope_records());
  REQUIRE(restored.abs_integral_estimate().count() == 64);
  original.adapt_envelope();
  restored.adapt_envelope();
  REQUIRE(factors(restored) == factors(original));
  REQUIRE(restored.abs_integral_estimate().count() == 64);
  int extra = 0;
  REQUIRE_NOTHROW(kakuhen::util::read_user_data(files.dir / "state.khs", extra, "extra"));
  REQUIRE(extra == 42);
  std::stringstream stream;
  auto pending = make_generator<TestType>();
  pending.collect_envelope(integrand, 8);
  pending.write_state_stream(stream);
  kakuhen::util::write_user_data_stream(stream, 42, "extra");
  restored.read_state_stream(stream);
  extra = 0;
  kakuhen::util::read_user_data_stream(stream, extra, "extra");
  REQUIRE(extra == 42);
  REQUIRE(restored.has_envelope_data());
}

TEMPLATE_TEST_CASE("Nonfinite collection and failed evaluations retain a consistent prefix",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  auto gen = make_generator<TestType>();
  const auto result = gen.collect_envelope(
      [](const Point<>& p) {
        return p.sample_index % 2 ? std::numeric_limits<double>::infinity() : 3.0 / p.weight;
      },
      8);
  REQUIRE(result.n_nonfinite() == 4);
  REQUIRE(result.n_violations() == 4);
  REQUIRE(result.n_evaluations() == 8);
  REQUIRE(gen.abs_integral_estimate().value() == Approx(1.5));
  REQUIRE_THROWS(gen.collect_envelope(
      [](const Point<>& p) {
        if (p.sample_index == 2) throw std::runtime_error("failed evaluation");
        return 3.0 / p.weight;
      },
      4));
  REQUIRE(gen.envelope_ready());
  REQUIRE(gen.n_envelope_records() == 6);
  REQUIRE(gen.abs_integral_estimate().count() == 10);
  const auto replay = gen.adapt_envelope();
  REQUIRE(replay.n_nonfinite() == 0);  // replay does not sample
  REQUIRE(replay.n_evaluations() == 10);
  REQUIRE(gen.abs_integral_estimate().count() == 10);
}

TEMPLATE_TEST_CASE("Changed maps invalidate pending envelope data", "[generator][envelope_data]",
                   VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-env-data");
  auto gen = make_generator<TestType>();
  gen.collect_envelope(integrand, 8);
  gen.grid_[0] *= 0.5;
  REQUIRE_FALSE(gen.envelope_ready());
  REQUIRE_THROWS_WITH(gen.adapt_envelope(),
                      Catch::Matchers::ContainsSubstring("clear_envelope_data()"));
  REQUIRE_THROWS(gen.collect_envelope(integrand, 8));
  REQUIRE_THROWS(gen.save_envelope(files.dir / "stale.khe"));
  // a stale envelope cannot be adapted, so the error must not suggest it
  REQUIRE_THROWS_WITH(gen.initialize_envelope(1.0),
                      Catch::Matchers::ContainsSubstring("not ready") &&
                          Catch::Matchers::ContainsSubstring("clear") &&
                          !Catch::Matchers::ContainsSubstring("adapt"));
  gen.clear_envelope_data();
  REQUIRE_THROWS_WITH(gen.adapt_envelope(), Catch::Matchers::ContainsSubstring("re-initialize") &&
                                                !Catch::Matchers::ContainsSubstring("clear"));
  REQUIRE_NOTHROW(gen.initialize_envelope(1.0));
}

TEMPLATE_TEST_CASE("Loading drops a pending batch whose envelope is not ready",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  auto gen = make_generator<TestType>();
  gen.collect_envelope(integrand, 8);
  gen.grid_[0] *= 0.5;  // the saved ready flag stays set, but the envelope is stale
  REQUIRE(gen.has_envelope_data());
  std::stringstream stream;
  gen.write_state_stream(stream);
  auto restored = make_generator<TestType>();
  restored.read_state_stream(stream);
  REQUIRE_FALSE(restored.envelope_ready());
  REQUIRE_FALSE(restored.has_envelope_data());
  REQUIRE_NOTHROW(restored.initialize_envelope(1.0));
}

TEMPLATE_TEST_CASE("Interrupted raising keeps the envelope and the pending batch",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  auto gen = make_generator<TestType>();
  gen.collect_envelope(integrand, 8);
  std::ostringstream before;
  gen.write_state_stream(before);
  unsigned calls = 0;
  REQUIRE_THROWS(gen.raise_envelope(
      [&](const Point<>& p) {
        if (++calls == 4) throw std::runtime_error("interrupted");
        return 10.0 / p.weight;  // violates everywhere, so the first samples raise
      },
      8));
  std::ostringstream after;
  gen.write_state_stream(after);
  REQUIRE(after.str() == before.str());
  REQUIRE(gen.envelope_ready());
  REQUIRE(gen.has_envelope_data());
  REQUIRE(gen.adapt_envelope().n_evaluations() == 8);
}

TEMPLATE_TEST_CASE("Overflowing statistics keep the valid prefix of a batch",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-env-data");
  auto gen = make_generator<TestType>();
  // each square is finite, but the sum of two is not
  auto huge = [](const Point<>& p) { return 1e154 / p.weight; };
  REQUIRE_THROWS_AS(gen.collect_envelope(huge, 2), std::overflow_error);
  REQUIRE(gen.envelope_ready());
  REQUIRE(gen.n_envelope_records() == 1);
  REQUIRE(gen.abs_integral_estimate().count() == 1);
  REQUIRE(gen.abs_integral_estimate().is_finite());
  gen.save_envelope(files.dir / "prefix.khe");
  auto receiver = make_generator<TestType>();
  REQUIRE_NOTHROW(receiver.merge_envelope(files.dir / "prefix.khe"));
  REQUIRE(receiver.abs_integral_estimate().count() == 1);
  REQUIRE(gen.adapt_envelope().n_evaluations() == 1);
}

TEST_CASE("Evaluation counts cannot wrap around", "[generator][envelope_data]") {
  // with a narrow count type, a wrapped batch count would hide the pending records
  TempDir files("kakuhen-env-data");
  VegasGenerator<kakuhen::util::num_traits_t<double, uint32_t, uint8_t>> gen(2, 4);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);
  auto above = [](const auto& p) { return 2.0 / p.weight; };
  gen.collect_envelope(above, 128);
  const auto before = snapshot(gen);
  REQUIRE_THROWS_AS(gen.collect_envelope(above, 128), std::overflow_error);
  REQUIRE_THROWS_AS(gen.raise_envelope(above, 128), std::overflow_error);
  // a rejected budget changes nothing, and the envelope stays usable
  REQUIRE(snapshot(gen) == before);
  REQUIRE(gen.envelope_ready());
  REQUIRE(gen.has_envelope_data());
  REQUIRE(gen.n_envelope_records() == 128);
  REQUIRE_NOTHROW(gen.collect_envelope(above, 127));
  REQUIRE_THROWS_AS(gen.raise_envelope(above, 1), std::overflow_error);
  REQUIRE(gen.envelope_ready());
  REQUIRE_NOTHROW(gen.save_envelope(files.dir / "narrow.khe"));
  REQUIRE(gen.adapt_envelope().n_evaluations() == 255);
  REQUIRE_FALSE(gen.has_envelope_data());
}

TEMPLATE_TEST_CASE("Malformed and truncated observation payloads leave imports unchanged",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-env-data");
  auto source = make_generator<TestType>();
  source.collect_envelope([](const Point<>& p) { return 100.0 / p.weight; }, 2);
  const auto path = files.dir / "batch.khe";
  source.save_envelope(path);
  const auto bytes = read_file(path);
  auto target = make_generator<TestType>();
  target.collect_envelope(integrand, 8);
  const auto before = snapshot(target);
  SECTION("every truncation") {
    for (std::size_t n = 0; n < bytes.size(); ++n) {
      write_file(path, bytes.substr(0, n));
      REQUIRE_THROWS(target.merge_envelope(path));
      REQUIRE(snapshot(target) == before);
    }
  }
  SECTION("out-of-range factor") {
    auto corrupt = bytes;
    std::ostringstream index;
    kakuhen::util::serialize::serialize_one(
        index, std::numeric_limits<typename TestType::size_type>::max());
    corrupt.replace(corrupt.size() - index.str().size(), index.str().size(), index.str());
    write_file(path, corrupt);
    REQUIRE_THROWS(target.merge_envelope(path));
    REQUIRE(snapshot(target) == before);
  }
  SECTION("wrong factor group") {
    auto corrupt = bytes;
    std::ostringstream index;
    kakuhen::util::serialize::serialize_one(index, typename TestType::size_type(0));
    corrupt.replace(corrupt.size() - index.str().size(), index.str().size(), index.str());
    write_file(path, corrupt);
    REQUIRE_THROWS(target.merge_envelope(path));
    REQUIRE(snapshot(target) == before);
  }
  SECTION("invalid batch fields") {
    using U = typename TestType::count_type;
    using S = typename TestType::size_type;
    const auto records_size = 2 * (sizeof(double) + source.ndim() * sizeof(S));
    const auto flag = bytes.size() - records_size - 2 * sizeof(U) - 2 * sizeof(double) - 1;
    const auto check = [&](std::size_t offset, auto value) {
      auto corrupt = bytes;
      std::ostringstream field;
      kakuhen::util::serialize::serialize_one(field, value);
      corrupt.replace(offset, field.str().size(), field.str());
      write_file(path, corrupt);
      REQUIRE_THROWS(target.merge_envelope(path));
      REQUIRE(snapshot(target) == before);
    };
    check(flag, uint8_t(2));
    check(flag + 1, -1.0);  // negative sum
    check(flag + 1, std::numeric_limits<double>::infinity());
    check(flag + 1 + sizeof(double), std::numeric_limits<double>::quiet_NaN());
    check(flag + 1 + 2 * sizeof(double), U(0));  // present batch without evaluations
    check(flag + 1 + 2 * sizeof(double) + sizeof(U), std::numeric_limits<U>::max());
    check(bytes.size() - records_size, -1.0);
    check(bytes.size() - records_size, std::numeric_limits<double>::infinity());
    check(bytes.size() - records_size, std::numeric_limits<double>::quiet_NaN());
  }
  write_file(path, bytes);
  REQUIRE_NOTHROW(target.merge_envelope(path));
  REQUIRE(target.n_envelope_records() > 0);
}

TEMPLATE_TEST_CASE("Import statistic overflow does not consume the receiver batch",
                   "[generator][envelope_data]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-env-data");
  auto source = make_generator<TestType>();
  auto target = make_generator<TestType>();
  auto huge = [](const Point<>& p) { return 1e153 / p.weight; };
  source.collect_envelope(huge, 100);
  target.collect_envelope(huge, 100);
  source.save_envelope(files.dir / "overflow.khe");
  const auto before = snapshot(target);
  REQUIRE_THROWS_AS(target.merge_envelope(files.dir / "overflow.khe"), std::overflow_error);
  REQUIRE(snapshot(target) == before);
  REQUIRE(target.n_envelope_records() == 100);
}

TEMPLATE_TEST_CASE("Merging reports the replayed batch", "[generator][envelope_data]",
                   VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-env-data");
  auto worker = make_generator<TestType>();
  auto twin = make_generator<TestType>();
  auto receiver = make_generator<TestType>();
  const auto with_nan = [](const Point<>& p) {
    return p.sample_index % 9 == 4 ? std::numeric_limits<double>::quiet_NaN() : integrand(p);
  };
  const auto collected = worker.collect_envelope(with_nan, 256);
  twin.collect_envelope(with_nan, 256);
  REQUIRE(collected.n_nonfinite() > 0);
  worker.save_envelope(files.dir / "batch.khe");

  // the receiver starts from the same envelope, so it replays like the twin adapts
  const auto merged = receiver.merge_envelope(files.dir / "batch.khe");
  const auto adapted = twin.adapt_envelope();
  REQUIRE(adapted.n_raised() > 0);
  REQUIRE(adapted.n_violations() == collected.n_violations());
  REQUIRE(merged.n_violations() == adapted.n_violations());
  REQUIRE(merged.n_raised() == adapted.n_raised());
  REQUIRE(merged.n_evaluations() == 256);
  REQUIRE(merged.n_nonfinite() == 0);  // merging does not sample
  REQUIRE(merged.count() == receiver.abs_integral_estimate().count());
  REQUIRE(merged.volume() == receiver.envelope_volume());

  twin.save_envelope(files.dir / "factors.khe");
  const auto factors_only = receiver.merge_envelope(files.dir / "factors.khe");
  REQUIRE(factors_only.n_violations() == 0);
  REQUIRE(factors_only.n_evaluations() == 0);
  REQUIRE(factors_only.volume() == receiver.envelope_volume());
}
