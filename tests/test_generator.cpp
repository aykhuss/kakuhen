#include "kakuhen/integrator/basin_generator.h"
#include "kakuhen/integrator/vegas_generator.h"
#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <sstream>
#include <vector>

using namespace kakuhen::integrator;
using Catch::Approx;

template <typename G>
concept ExposesGeneratorStorage = requires(G& g) { g.ndim_; };
static_assert(!ExposesGeneratorStorage<VegasGenerator<>>);
static_assert(!ExposesGeneratorStorage<BasinGenerator<>>);

namespace {
/// a two-dimensional generator with a grid fine enough for statistical checks
template <typename G>
G make_generator_2d() {
  if constexpr (std::same_as<G, BasinGenerator<>>) {
    return G(2, 4, 8);
  } else {
    return G(2, 16);
  }
}
}  // namespace

TEMPLATE_TEST_CASE("Generators enforce frozen grid preconditions", "[generator]", VegasGenerator<>,
                   BasinGenerator<>) {
  TestType gen(1);
  auto integrand = [](const Point<>&) { return 1.0; };
  auto callback = [](const Point<>&, double) {};

  REQUIRE_THROWS_AS(gen.initialize_envelope(1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(gen.initialize_envelope(integrand, 8ULL), std::invalid_argument);
  REQUIRE_THROWS_AS(gen.optimize_envelope(integrand, 8), std::invalid_argument);
  REQUIRE_THROWS_AS(gen.generate_trials(integrand, 1, callback), std::invalid_argument);

  gen.set_options({.frozen = true, .verbosity = 0});

  REQUIRE_THROWS_AS(gen.initialize_envelope(integrand, 0ULL), std::invalid_argument);
  REQUIRE_THROWS_AS(gen.generate_trials(integrand, 0, callback), std::invalid_argument);
}

TEMPLATE_TEST_CASE("Envelope seeds must be finite and positive", "[generator]", VegasGenerator<>,
                   BasinGenerator<>) {
  TestType gen(2);
  gen.set_options({.frozen = true, .verbosity = 0});
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const double seed : {0.0, -0.0, -1.0, inf, nan}) {
    REQUIRE_THROWS_AS(gen.initialize_envelope(seed), std::invalid_argument);
    REQUIRE_FALSE(gen.envelope_ready());
  }
  auto zero = [](const Point<>&) { return 0.0; };
  REQUIRE_THROWS_AS(gen.initialize_envelope(zero, 100ULL), std::runtime_error);
  REQUIRE_FALSE(gen.envelope_ready());
}

TEMPLATE_TEST_CASE("Generators generate with an explicitly initialized envelope", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  TestType gen(2);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);
  REQUIRE(gen.envelope_ready());
  REQUIRE(gen.envelope_seed() == Approx(1.0));
  REQUIRE(gen.envelope_volume() == Approx(1.0));

  auto integrand = [](const Point<>&) { return 1.0; };
  int callback_count = 0;
  double weight_sum = 0.0;
  auto callback = [&](const Point<>&, double event_weight) {
    ++callback_count;
    weight_sum += event_weight;
  };

  auto result = gen.generate_trials(integrand, 5, callback);

  REQUIRE(callback_count == 5);
  REQUIRE(weight_sum == Approx(5.0));
  REQUIRE(result.status() == GenerationStatus::COMPLETED);
  REQUIRE(result.n_events() == 5);
  // flat unit integrand against a unit envelope: every trial accepted
  REQUIRE(result.n_trials() == 5);
  REQUIRE(result.n_rejected() == 0);
  REQUIRE(result.n_overweight() == 0);
  REQUIRE(result.n_negative() == 0);
  REQUIRE(result.volume() == Approx(1.0));
  // V_B * sum(s*w)/n_trials reproduces the integral exactly here
  REQUIRE(result.value() == Approx(1.0));
}

TEMPLATE_TEST_CASE("Generator event sum reproduces the integral", "[generator]", VegasGenerator<>,
                   BasinGenerator<>) {
  auto gen = make_generator_2d<TestType>();
  // true integral = 1
  auto integrand = [](const Point<>& point) { return 4.0 * point.x[0] * point.x[1]; };

  gen.integrate(integrand,
                {.neval = 5000, .niter = 4, .adapt = true, .verbosity = 0, .progress_bar = false});
  gen.set_options({.frozen = true, .verbosity = 0});
  // the seed comes from a frozen production run (the canonical workflow)
  auto prod =
      gen.integrate(integrand, {.neval = 5000, .niter = 1, .verbosity = 0, .progress_bar = false});
  gen.initialize_envelope(std::abs(prod.value()));
  auto env = gen.optimize_envelope(integrand, 20000);

  REQUIRE(env.abs_integral() == Approx(1.0).margin(5.0 * env.abs_error()));
  REQUIRE(env.efficiency() > 0.0);
  REQUIRE(gen.predicted_efficiency() == Approx(env.efficiency()));

  double weight_sum = 0.0;
  auto callback = [&](const Point<>&, double event_weight) { weight_sum += event_weight; };
  auto result = gen.generate_trials(integrand, 20000, callback);

  REQUIRE(result.status() == GenerationStatus::COMPLETED);
  REQUIRE(result.n_trials() == 20000);
  REQUIRE(result.n_events() > 0);
  // unnormalized +-1 events: the user-side normalization is V_B / n_trials
  const double estimate = result.volume() * weight_sum / static_cast<double>(result.n_trials());
  REQUIRE(estimate == Approx(1.0).margin(5.0 * result.error()));
  // ... which is exactly what result.value() reports
  REQUIRE(result.value() == Approx(estimate));
  // ... and normalization() is that per-event factor
  REQUIRE(result.normalization() ==
          Approx(result.volume() / static_cast<double>(result.n_trials())));
  // positive integrand: no negative events
  REQUIRE(result.n_negative() == 0);
}

TEMPLATE_TEST_CASE("Generator state stream round-trips the envelope", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  auto gen = make_generator_2d<TestType>();
  // ridge integrand so the BASIN round-trip covers conditional envelope tables
  auto integrand = [](const Point<>& point) {
    const double d = point.x[0] - point.x[1];
    return std::exp(-100.0 * d * d);
  };

  gen.integrate(integrand,
                {.neval = 20000, .niter = 5, .adapt = true, .verbosity = 0, .progress_bar = false});
  gen.set_options({.frozen = true, .verbosity = 0});
  auto prod =
      gen.integrate(integrand, {.neval = 20000, .niter = 1, .verbosity = 0, .progress_bar = false});
  if constexpr (std::same_as<TestType, BasinGenerator<>>) {
    REQUIRE(gen.nblocks() < 2);
  }
  gen.initialize_envelope(std::abs(prod.value()));
  gen.optimize_envelope(integrand, 10000);

  std::stringstream ss;
  gen.write_state_stream(ss);

  // a worker restores grid + envelope and can generate immediately
  auto worker = make_generator_2d<TestType>();
  worker.set_options({.frozen = true, .verbosity = 0});
  worker.read_state_stream(ss);

  REQUIRE(worker.hash().value() == gen.hash().value());
  REQUIRE(worker.envelope_ready());
  REQUIRE(worker.envelope_seed() == Approx(gen.envelope_seed()));
  REQUIRE(worker.envelope_volume() == Approx(gen.envelope_volume()));
  REQUIRE(worker.abs_integral_estimate().count() == gen.abs_integral_estimate().count());
  REQUIRE(worker.abs_integral_estimate().value() == Approx(gen.abs_integral_estimate().value()));

  auto callback = [](const Point<>&, double) {};
  worker.set_seed(7);
  auto res_worker = worker.generate_trials(integrand, 10000, callback);
  auto res_gen = gen.generate_trials(integrand, 10000, callback);

  // both runs share one envelope: their results merge
  auto merged = decltype(res_gen){};
  merged.accumulate(res_worker);
  merged.accumulate(res_gen);
  REQUIRE(merged.status() == GenerationStatus::COMPLETED);
  const double sigma = std::sqrt(merged.error() * merged.error() + prod.error() * prod.error());
  REQUIRE(merged.value() == Approx(prod.value()).margin(5.0 * sigma));
}

TEMPLATE_TEST_CASE("Integration checkpoints keep the file consistent with the envelope",
                   "[generator]", VegasGenerator<>, BasinGenerator<>) {
  auto gen = make_generator_2d<TestType>();
  auto integrand = [](const Point<>& point) { return 1.0 + 10.0 * point.x[0]; };
  const std::filesystem::path fpath =
      "test_generator_checkpoint_" + std::string(to_string(TestType::class_id())) + ".khs";

  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);
  gen.optimize_envelope(integrand, 256);
  gen.save(fpath);

  auto reload = [&] {
    auto other = make_generator_2d<TestType>();
    other.set_options({.frozen = true, .verbosity = 0});
    other.load(fpath);
    return other;
  };

  SECTION("production on the frozen grid preserves the envelope block") {
    gen.integrate(
        integrand,
        {.neval = 1024, .niter = 2, .verbosity = 0, .file_path = fpath, .progress_bar = false});
    auto other = reload();
    REQUIRE(other.envelope_ready());
    REQUIRE(other.envelope_volume() == gen.envelope_volume());
  }

  SECTION("adaptation checkpoints the new grid and drops the stale envelope") {
    gen.set_options({.frozen = false, .verbosity = 0});
    gen.integrate(integrand, {.neval = 1024,
                              .niter = 2,
                              .adapt = true,
                              .verbosity = 0,
                              .file_path = fpath,
                              .progress_bar = false});
    gen.set_options({.frozen = true, .verbosity = 0});
    auto other = reload();
    REQUIRE(other.hash().value() == gen.hash().value());
    REQUIRE_FALSE(gen.envelope_ready());
    REQUIRE_FALSE(other.envelope_ready());
  }

  std::filesystem::remove(fpath);
}

TEMPLATE_TEST_CASE("Generators reject generation from a stale envelope", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  auto gen = make_generator_2d<TestType>();
  auto integrand = [](const Point<>& point) { return 1.0 + 10.0 * point.x[0]; };
  auto callback = [](const Point<>&, double) {};

  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);
  gen.optimize_envelope(integrand, 64);
  REQUIRE(gen.envelope_ready());
  const auto envelope_grid_hash = gen.hash().value();

  gen.set_options({.frozen = false, .verbosity = 0});
  gen.integrate(integrand,
                {.neval = 2048, .niter = 3, .adapt = true, .verbosity = 0, .progress_bar = false});
  REQUIRE(gen.hash().value() != envelope_grid_hash);

  gen.set_options({.frozen = true, .verbosity = 0});
  REQUIRE_FALSE(gen.envelope_ready());
  REQUIRE_THROWS_AS(gen.optimize_envelope(integrand, 64), std::runtime_error);
  REQUIRE_THROWS_AS(gen.generate_trials(integrand, 1, callback), std::runtime_error);
  REQUIRE_THROWS_AS(gen.predicted_efficiency(), std::runtime_error);
  // scaling must not re-bind a stale envelope to the new grid
  REQUIRE_THROWS_AS(gen.envelope_scale(1.0), std::runtime_error);
  REQUIRE_THROWS_AS(gen.generate_trials(integrand, 1, callback), std::runtime_error);
}

TEMPLATE_TEST_CASE("Failed envelope updates require reinitialization", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  TestType gen(1);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);
  unsigned calls = 0;
  SECTION("integrand throws after raising a cell") {
    auto fail = [&](const Point<>&) {
      if (++calls == 2) throw std::runtime_error("integrand failed");
      return 2.0;
    };
    REQUIRE_THROWS_AS(gen.raise_envelope(fail, 10), std::runtime_error);
  }
  SECTION("strict finite check throws after raising a cell") {
    gen.set_options({.strict_finite_integrand = true});
    auto fail = [&](const Point<>&) {
      return ++calls == 2 ? std::numeric_limits<double>::infinity() : 2.0;
    };
    REQUIRE_THROWS_AS(gen.raise_envelope(fail, 10), std::runtime_error);
  }
  SECTION("scaling overflows during sealing") {
    REQUIRE_THROWS(gen.envelope_scale(std::numeric_limits<double>::max()));
  }
  REQUIRE_FALSE(gen.envelope_ready());
  auto one = [](const Point<>&) { return 1.0; };
  auto sink = [](const Point<>&, double) {};
  REQUIRE_THROWS(gen.generate_trials(one, 10, sink));
  gen.initialize_envelope(1.0);
  REQUIRE(gen.generate_trials(one, 10, sink).value() == Approx(1.0));
}

TEMPLATE_TEST_CASE("Plain state loads resize all generator storage", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  typename TestType::IntBase plain(3);
  std::stringstream state;
  plain.write_state_stream(state);
  TestType worker(1, 2);
  worker.set_options({.frozen = true, .verbosity = 0});
  worker.initialize_envelope(1.0);
  worker.read_state_stream(state);
  REQUIRE(worker.ndim() == 3);
  REQUIRE_FALSE(worker.envelope_ready());
  worker.initialize_envelope(1.0);
  auto result = worker.generate_trials(
      [](const Point<>& p) {
        REQUIRE(p.x.size() == 3);
        return 1.0;
      },
      10, [](const Point<>&, double) {});
  REQUIRE(result.value() == Approx(1.0));
}

TEMPLATE_TEST_CASE("Unrepresentable weights fail before delivery", "[generator]", VegasGenerator<>,
                   BasinGenerator<>) {
  TestType gen(1);
  gen.set_options({.frozen = true, .verbosity = 0});
  bool delivered = false;
  auto sink = [&](const Point<>&, double) { delivered = true; };
  SECTION("ratio overflow from a tiny seed") {
    gen.initialize_envelope(1e-310);
  }
  SECTION("finite weight with an overflowing square") {
    gen.initialize_envelope(1e-200);
  }
  auto hundred = [](const Point<>&) { return 100.0; };
  REQUIRE_THROWS_AS(gen.generate_trials(hundred, 2, sink), std::overflow_error);
  REQUIRE_FALSE(delivered);
  gen.initialize_envelope(hundred, 10);
  REQUIRE(gen.generate_trials(hundred, 2, sink).value() == Approx(100.0));
}

TEMPLATE_TEST_CASE("Trial generation handles rejection and stopping", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  TestType gen(1);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.set_seed(42);
  gen.initialize_envelope(1.0);
  auto half = [](const Point<>&) { return 0.5; };
  auto sink = [](const Point<>&, double) {};
  REQUIRE_THROWS_AS(gen.generate_trials(half, 0, sink), std::invalid_argument);
  auto result = gen.generate_trials(half, 20000, sink);
  REQUIRE(result.status() == GenerationStatus::COMPLETED);
  REQUIRE(result.n_trials() == 20000);
  REQUIRE(result.n_events() < result.n_trials());
  REQUIRE(result.value() == Approx(0.5).margin(5 * result.error()));
  auto empty = gen.generate_trials([](const Point<>&) { return 0.0; }, 10, sink);
  REQUIRE(empty.status() == GenerationStatus::COMPLETED);
  REQUIRE(empty.n_trials() == 10);
  REQUIRE(empty.n_events() == 0);
  REQUIRE(empty.value() == 0.0);
  auto stopped =
      gen.generate_trials(half, 100, [](const Point<>&, double) { return EventSignal::STOP; });
  REQUIRE(stopped.status() == GenerationStatus::STOPPED);
  REQUIRE(stopped.n_events() == 1);
  auto merged = decltype(result){};
  merged.accumulate(result);
  merged.accumulate(empty);
  REQUIRE(merged.status() == GenerationStatus::COMPLETED);
  merged.accumulate(stopped);
  REQUIRE(merged.status() == GenerationStatus::STOPPED);
  REQUIRE(merged.n_trials() == result.n_trials() + empty.n_trials() + stopped.n_trials());
  // an empty stopped result must still taint the merged status
  auto tainted = decltype(result){};
  tainted.accumulate(result);
  auto stopped_empty = decltype(result){};
  stopped_empty.status_ = GenerationStatus::STOPPED;
  tainted.accumulate(stopped_empty);
  REQUIRE(tainted.status() == GenerationStatus::STOPPED);
  REQUIRE(tainted.n_trials() == result.n_trials());
}

TEST_CASE("Stopping after the first event biases the estimate", "[generator]") {
  VegasGenerator<> gen(1, 4);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.set_seed(42);
  gen.initialize_envelope(1.0);
  auto half = [](const Point<>&) { return 0.5; };
  auto sink = [](const Point<>&, double) {};
  auto stop = [](const Point<>&, double) { return EventSignal::STOP; };
  double fixed_sum = 0;
  double stopped_sum = 0;
  for (unsigned i = 0; i < 20000; ++i) {
    fixed_sum += gen.generate_trials(half, 1, sink).value();
    // efficiency 1/2: the budget cannot realistically run out before the first event
    stopped_sum += gen.generate_trials(half, 1000, stop).value();
  }
  REQUIRE(fixed_sum / 20000 == Approx(0.5).margin(0.015));
  REQUIRE(stopped_sum / 20000 == Approx(std::log(2.0)).margin(0.015));
}

TEST_CASE("Branchless CDF search agrees with std::upper_bound", "[generator]") {
  for (std::size_t size = 1; size <= 9; ++size) {
    std::vector<double> cdf(size);
    for (std::size_t i = 0; i < size; ++i)
      cdf[i] = double(i + 1);
    for (double target = -0.5; target <= double(size) + 0.5; target += 0.25) {
      CAPTURE(size, target);
      const auto expected = std::upper_bound(cdf.begin(), cdf.end(), target) - cdf.begin();
      REQUIRE(detail::upper_bound_branchless<double>(cdf, target) ==
              static_cast<std::size_t>(expected));
    }
  }
}

TEST_CASE("Envelope CDF inversion stays inside cells at boundaries", "[generator]") {
  const std::array<double, 3> cdf{1, 3, 6};
  const std::uint32_t expected_cell[] = {0, 1, 2, 2, 2};
  unsigned i = 0;
  for (double u : {0.0, 1.0 / 6, 0.5, std::nextafter(1.0, 0.0), 1.0}) {
    const auto draw = detail::sample_cdf<double, std::uint32_t>(cdf, u);
    REQUIRE(draw.cell == expected_cell[i++]);
    REQUIRE(draw.fraction >= 0.0);
    REQUIRE(draw.fraction <= 1.0);
    REQUIRE(draw.width == double(draw.cell + 1));
  }
  REQUIRE_THROWS(detail::cdf_step(1.0, -1.0));
  REQUIRE_THROWS(detail::cdf_step(1.0, 0.0));
  REQUIRE_THROWS(detail::cdf_step(1.0, 1e-20));
}

TEMPLATE_TEST_CASE("Generator loads reject invalid positive-volume envelopes", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  TestType gen(1, 2);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);
  std::stringstream state;
  gen.write_state_stream(state);
  // The final two raw doubles belong to a CDF row. Their sum stays positive.
  std::string bytes = state.str();
  std::ostringstream replacement;
  kakuhen::util::serialize::serialize_one(replacement, -1.0);
  kakuhen::util::serialize::serialize_one(replacement, 3.0);
  bytes.replace(bytes.size() - replacement.str().size(), replacement.str().size(),
                replacement.str());
  std::istringstream corrupt(bytes);
  REQUIRE_THROWS_AS(gen.read_state_stream(corrupt), std::runtime_error);
  REQUIRE_FALSE(gen.envelope_ready());
}

TEMPLATE_TEST_CASE("Accumulated overweight squares cannot silently overflow", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  TestType gen(1);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1e-154);
  unsigned delivered = 0;
  REQUIRE_THROWS_AS(gen.generate_trials([](const Point<>&) { return 1.0; }, 3,
                                        [&](const Point<>&, double w) {
                                          REQUIRE(std::isfinite(w));
                                          ++delivered;
                                        }),
                    std::overflow_error);
  // each weight 1e154 and its square are finite, so all events are delivered;
  // the overflowing sum of squares is caught when the run ends
  REQUIRE(delivered == 3);
}

TEMPLATE_TEST_CASE("The generation progress bar does not change the run", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  auto half = [](const Point<>& point) { return point.x[0]; };
  struct Run {
    std::vector<double> weights;
    GenerationResult<double, unsigned long long> result;
  };
  auto run = [&](bool show_bar, bool stop) {
    TestType gen(1);
    gen.set_options({.frozen = true, .verbosity = show_bar ? 1 : 0, .progress_bar = show_bar});
    gen.set_seed(7);
    gen.initialize_envelope(1.0);
    Run r;
    auto sink = [&](const Point<>&, double w) {
      r.weights.push_back(w);
      return stop && r.weights.size() >= 500 ? EventSignal::STOP : EventSignal::NONE;
    };
    r.result = gen.generate_trials(half, 2000, sink);
    return r;
  };
  for (const bool stop : {false, true}) {
    const Run quiet = run(false, stop);
    const Run shown = run(true, stop);
    REQUIRE((quiet.result.status() == GenerationStatus::STOPPED) == stop);
    REQUIRE(quiet.weights == shown.weights);
    REQUIRE(quiet.result.n_trials() == shown.result.n_trials());
    REQUIRE(quiet.result.n_events() == shown.result.n_events());
    REQUIRE(quiet.result.status() == shown.result.status());
    REQUIRE(quiet.result.value() == shown.result.value());
    REQUIRE(quiet.result.error() == shown.result.error());
  }
}

TEMPLATE_TEST_CASE("The generation progress bar validates progress_step", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  TestType gen(1);
  gen.set_options({.frozen = true, .verbosity = 0, .progress_step = 0.0});
  gen.initialize_envelope(1.0);
  auto one = [](const Point<>&) { return 1.0; };
  auto sink = [](const Point<>&, double) {};
  // without the bar, the step is irrelevant
  REQUIRE_NOTHROW(gen.generate_trials(one, 1, sink));
  gen.set_options({.verbosity = 1, .progress_bar = true});
  REQUIRE_THROWS_AS(gen.generate_trials(one, 1, sink), std::invalid_argument);
  gen.set_options({.progress_step = 1.5});
  REQUIRE_THROWS_AS(gen.generate_trials(one, 1, sink), std::invalid_argument);
}

TEMPLATE_TEST_CASE("predicted_efficiency prefers the sampled abs-integral", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  TestType gen(1);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.set_seed(42);
  REQUIRE_THROWS_AS(gen.predicted_efficiency(), std::runtime_error);  // no envelope
  // without a sampled estimate the seed stands in for A: a flat envelope predicts 1
  gen.initialize_envelope(0.5);
  REQUIRE(gen.predicted_efficiency() == Approx(1.0));
  gen.envelope_scale(4.0);
  REQUIRE(gen.predicted_efficiency() == Approx(0.25));
  // a sampled estimate takes precedence over the seed
  auto half = [](const Point<>&) { return 0.5; };
  gen.initialize_envelope(half, 1000ULL);
  gen.envelope_scale(2.0);
  REQUIRE(gen.predicted_efficiency() == Approx(0.5));
  auto result = gen.generate_trials(half, 20000, [](const Point<>&, double) {});
  REQUIRE(result.efficiency() == Approx(0.5).margin(0.02));
}

TEMPLATE_TEST_CASE("Absolute-integral estimation rejects overflowing squares", "[generator]",
                   VegasGenerator<>, BasinGenerator<>) {
  TestType gen(1);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);
  REQUIRE_THROWS_AS(gen.initialize_envelope([](const Point<>&) { return 1e200; }, 2),
                    std::overflow_error);
  REQUIRE_FALSE(gen.envelope_ready());
}
