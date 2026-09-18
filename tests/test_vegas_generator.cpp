#include "kakuhen/integrator/vegas.h"
#include "kakuhen/integrator/vegas_generator.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace kakuhen::integrator;
using Catch::Approx;

TEST_CASE("VegasGenerator requires an initialized envelope", "[vegas_generator]") {
  VegasGenerator<> gen(1, 4);
  gen.set_options({.frozen = true, .verbosity = 0});

  auto integrand = [](const Point<>&) { return 1.0; };
  auto callback = [](const Point<>&, double) {};

  // optimize before initialize and generate before initialize both throw
  REQUIRE_THROWS_AS(gen.optimize_envelope(integrand, 8), std::runtime_error);
  REQUIRE_THROWS_AS(gen.generate_trials(integrand, 1, callback), std::runtime_error);
  REQUIRE_FALSE(gen.envelope_ready());
}

TEST_CASE("VegasGenerator seeds the envelope from a production integration result",
          "[vegas_generator]") {
  VegasGenerator<> gen(2, 8);
  auto integrand = [](const Point<>& point) { return 1.0 + point.x[0] + point.x[1]; };

  gen.integrate(integrand,
                {.neval = 2000, .niter = 3, .adapt = true, .verbosity = 0, .progress_bar = false});
  gen.set_options({.frozen = true, .verbosity = 0});
  // the seed comes from a frozen production run (the canonical workflow)
  auto prod =
      gen.integrate(integrand, {.neval = 2000, .niter = 1, .verbosity = 0, .progress_bar = false});
  gen.initialize_envelope(std::abs(prod.value()));

  REQUIRE(gen.envelope_ready());
  // seed = |I| of the production run; the true integral is 2
  REQUIRE(gen.envelope_seed() == Approx(2.0).margin(0.2));
  REQUIRE(gen.envelope_volume() == Approx(gen.envelope_seed()));
  // the from-result seed is not a sampled A estimate
  REQUIRE(gen.abs_integral_estimate().count() == 0);
}

TEST_CASE("VegasGenerator raise_envelope raises the envelope and reports diagnostics",
          "[vegas_generator]") {
  VegasGenerator<> gen(2, 16);
  // positive integrand, true integral = 4 * 1/2 * 1/2 = 1
  auto integrand = [](const Point<>& point) { return 4.0 * point.x[0] * point.x[1]; };

  gen.integrate(integrand,
                {.neval = 5000, .niter = 4, .adapt = true, .verbosity = 0, .progress_bar = false});
  gen.set_options({.frozen = true, .verbosity = 0});
  auto prod =
      gen.integrate(integrand, {.neval = 5000, .niter = 1, .verbosity = 0, .progress_bar = false});
  gen.initialize_envelope(std::abs(prod.value()));
  const double seed_volume = gen.envelope_volume();

  auto env = gen.raise_envelope(integrand, 20000);

  // the A estimate matches the abs-integral (= integral for positive f)
  REQUIRE(env.abs_integral() == Approx(1.0).margin(5.0 * env.abs_error()));
  REQUIRE(env.count() == 20000);
  // raising can only grow the envelope
  REQUIRE(env.volume() >= seed_volume);
  REQUIRE(env.volume() == Approx(gen.envelope_volume()));
  REQUIRE(env.efficiency() > 0.0);
  REQUIRE(env.efficiency() <= 1.0);

  // repeated passes accumulate A statistics and converge (fewer violations)
  auto env2 = gen.raise_envelope(integrand, 20000);
  REQUIRE(env2.count() == 40000);
  REQUIRE(env2.n_violations() <= env.n_violations());
}

TEST_CASE("VegasGenerator optimize_envelope drives raising passes to convergence",
          "[vegas_generator]") {
  VegasGenerator<> gen(1, 4);
  gen.set_options({.frozen = true, .verbosity = 0});
  auto integrand = [](const Point<>&) { return 1.0; };

  SECTION("stops early once the violation rate meets the target") {
    // a unit envelope is never violated by a unit integrand: one pass suffices
    gen.initialize_envelope(1.0);
    auto env = gen.optimize_envelope(integrand, 64, 8);
    REQUIRE(env.count() == 64);
    REQUIRE(env.n_violations() == 0);
  }

  SECTION("exhausts the pass budget while violations persist") {
    // a tiny seed needs many raises to reach the integrand: every pass violates
    gen.initialize_envelope(1e-6);
    auto env = gen.optimize_envelope(integrand, 64, 3);
    REQUIRE(env.count() == 3 * 64);
    REQUIRE(env.n_violations() > 0);
  }

  SECTION("rejects an empty pass budget") {
    gen.initialize_envelope(1.0);
    REQUIRE_THROWS_AS(gen.optimize_envelope(integrand, 64, 0), std::invalid_argument);
  }
}

TEST_CASE("VegasGenerator handles sign-changing integrands", "[vegas_generator]") {
  VegasGenerator<> gen(1, 16);
  // f = sin(2*pi*x): integral 0, abs-integral 2/pi
  auto integrand = [](const Point<>& point) { return std::sin(2.0 * M_PI * point.x[0]); };

  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(integrand, 5000ULL);
  gen.optimize_envelope(integrand, 20000);

  auto callback = [](const Point<>&, double) {};
  auto result = gen.generate_trials(integrand, 30000, callback);

  REQUIRE(result.status() == GenerationStatus::COMPLETED);
  REQUIRE(result.n_negative() > 0);
  REQUIRE(result.negative_fraction() == Approx(0.5).margin(0.05));
  REQUIRE(result.value() == Approx(0.0).margin(5.0 * result.error()));
  // A diagnostic: 2/pi
  REQUIRE(gen.abs_integral_estimate().value() ==
          Approx(2.0 / M_PI).margin(5.0 * gen.abs_integral_estimate().error()));
}

TEST_CASE("VegasGenerator stays unbiased with a deliberately tiny envelope", "[vegas_generator]") {
  VegasGenerator<> gen(1, 8);
  auto integrand = [](const Point<>& point) { return 1.0 + point.x[0]; };  // integral 1.5

  gen.set_options({.frozen = true, .verbosity = 0});
  // seed far below the true bound and skip optimization: most accepted events
  // are overweights, but the V_B / n_trials estimator remains unbiased
  gen.initialize_envelope(0.1);

  auto callback = [](const Point<>&, double) {};
  auto result = gen.generate_trials(integrand, 5000, callback);

  REQUIRE(result.status() == GenerationStatus::COMPLETED);
  REQUIRE(result.n_overweight() > 0);
  REQUIRE(result.max_overweight() > 1.0);
  REQUIRE(result.value() == Approx(1.5).margin(5.0 * result.error()));
}

TEST_CASE("VegasGenerator completes the trial budget for a zero integrand", "[vegas_generator]") {
  VegasGenerator<> gen(1, 4);
  gen.set_options({.frozen = true, .verbosity = 0});

  auto zero_integrand = [](const Point<>&) { return 0.0; };
  gen.initialize_envelope(1.0);

  bool called = false;
  auto callback = [&](const Point<>&, double) { called = true; };

  auto result = gen.generate_trials(zero_integrand, 16, callback);

  REQUIRE_FALSE(called);
  REQUIRE(result.status() == GenerationStatus::COMPLETED);
  REQUIRE(result.n_events() == 0);
  REQUIRE(result.n_trials() == 16);
  // a zero integrand yields a zero estimate, not a crash
  REQUIRE(result.value() == Approx(0.0));
}

TEST_CASE("VegasGenerator skips non-finite generation trials", "[vegas_generator]") {
  VegasGenerator<> gen(1, 4);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);

  auto integrand = [](const Point<>& point) {
    return point.sample_index < 2 ? std::numeric_limits<double>::quiet_NaN() : 1.0;
  };
  int callback_count = 0;
  auto callback = [&](const Point<>&, double event_weight) {
    ++callback_count;
    REQUIRE(event_weight == Approx(1.0));
  };

  auto result = gen.generate_trials(integrand, 5, callback);

  REQUIRE(callback_count == 3);
  REQUIRE(result.status() == GenerationStatus::COMPLETED);
  REQUIRE(result.n_events() == 3);
  REQUIRE(result.n_trials() == 5);
  REQUIRE(result.n_rejected() == 2);
  REQUIRE(result.n_nonfinite() == 2);
  REQUIRE(result.value() == Approx(0.6));
}

TEST_CASE("VegasGenerator skips non-finite envelope samples", "[vegas_generator]") {
  VegasGenerator<> gen(1, 4);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);

  auto integrand = [](const Point<>& point) {
    return point.sample_index % 2 == 0 ? std::numeric_limits<double>::quiet_NaN() : 1.0;
  };

  auto env = gen.raise_envelope(integrand, 4);

  REQUIRE(env.count() == 4);
  REQUIRE(env.n_nonfinite() == 2);
  REQUIRE(env.abs_integral() == Approx(0.5));
  REQUIRE(env.n_violations() == 0);
  REQUIRE(gen.abs_integral_estimate().count() == 4);
  REQUIRE(gen.abs_integral_estimate().value() == Approx(0.5));
}

TEST_CASE("VegasGenerator can throw on non-finite integrand values", "[vegas_generator]") {
  VegasGenerator<> gen(1, 4);
  gen.set_options({.frozen = true, .verbosity = 0, .strict_finite_integrand = true});
  gen.initialize_envelope(1.0);

  auto integrand = [](const Point<>&) { return std::numeric_limits<double>::quiet_NaN(); };
  auto callback = [](const Point<>&, double) {};

  REQUIRE_THROWS_AS(gen.optimize_envelope(integrand, 1), std::runtime_error);
  REQUIRE_THROWS_AS(gen.generate_trials(integrand, 16, callback), std::runtime_error);
}

TEST_CASE("VegasGenerator callback can stop generation", "[vegas_generator]") {
  VegasGenerator<> gen(1, 4);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);

  auto integrand = [](const Point<>&) { return 1.0; };
  int callback_count = 0;
  auto callback = [&](const Point<>&, double) -> EventSignal {
    ++callback_count;
    return callback_count >= 2 ? EventSignal::STOP : EventSignal::NONE;
  };

  auto result = gen.generate_trials(integrand, 100, callback);

  REQUIRE(callback_count == 2);
  REQUIRE(result.status() == GenerationStatus::STOPPED);
  REQUIRE(result.n_events() == 2);
  // a unit integrand under a unit envelope accepts every trial
  REQUIRE(result.n_trials() == 2);
}

TEST_CASE("VegasGenerator sample_index is the running trial index", "[vegas_generator]") {
  VegasGenerator<> gen(1, 8);
  // ~50% acceptance so trials > events
  auto integrand = [](const Point<>& point) { return point.x[0]; };

  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(1.0);

  Point<>::count_type last_index = 0;
  bool first = true;
  bool strictly_increasing = true;
  auto callback = [&](const Point<>& point, double) {
    if (!first && point.sample_index <= last_index) strictly_increasing = false;
    last_index = point.sample_index;
    first = false;
  };

  auto result = gen.generate_trials(integrand, 200, callback);

  REQUIRE(strictly_increasing);
  REQUIRE(result.n_trials() > result.n_events());
  // indices enumerate trials, so the last accepted index fits in [0, n_trials)
  REQUIRE(last_index < result.n_trials());
}

TEST_CASE("VegasGenerator reads plain integrator state without an envelope block",
          "[vegas_generator]") {
  auto integrand = [](const Point<>& point) { return 1.0 + point.x[0]; };

  Vegas<> plain(2, 16);
  plain.set_options({.verbosity = 0});
  plain.integrate(
      integrand, {.neval = 2000, .niter = 3, .adapt = true, .verbosity = 0, .progress_bar = false});

  std::stringstream ss;
  plain.write_state_stream(ss);

  VegasGenerator<> gen(2, 16);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.read_state_stream(ss);

  // the grid arrived, the envelope did not
  REQUIRE(gen.hash().value() == plain.hash().value());
  REQUIRE_FALSE(gen.envelope_ready());

  // ... and the other direction: a plain integrator reads the grid portion of
  // a generator stream, ignoring the trailing envelope block
  gen.initialize_envelope(1.0);
  std::stringstream ss_gen;
  gen.write_state_stream(ss_gen);
  Vegas<> plain2(2, 16);
  plain2.set_options({.verbosity = 0});
  plain2.read_state_stream(ss_gen);
  REQUIRE(plain2.hash().value() == gen.hash().value());
}

TEST_CASE("VegasGenerator save/load round-trips through a file", "[vegas_generator]") {
  auto integrand = [](const Point<>& point) { return 1.0 + point.x[0]; };
  const std::filesystem::path fpath = "test_vegas_generator_roundtrip.khs";

  VegasGenerator<> gen(1, 8);
  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(integrand, 2000ULL);
  gen.optimize_envelope(integrand, 5000);
  gen.save(fpath);

  VegasGenerator<> other(1, 8);
  other.set_options({.frozen = true, .verbosity = 0});
  other.load(fpath);
  std::filesystem::remove(fpath);

  REQUIRE(other.envelope_ready());
  REQUIRE(other.envelope_volume() == Approx(gen.envelope_volume()));

  auto callback = [](const Point<>&, double) {};
  auto result = other.generate_trials(integrand, 2000, callback);
  REQUIRE(result.status() == GenerationStatus::COMPLETED);
  REQUIRE(result.value() == Approx(1.5).margin(5.0 * result.error()));
}

TEST_CASE("GenerationResult merges across runs", "[vegas_generator]") {
  VegasGenerator<> gen(1, 16);
  auto integrand = [](const Point<>& point) { return 1.0 + point.x[0]; };  // integral 1.5

  gen.set_options({.frozen = true, .verbosity = 0});
  gen.initialize_envelope(integrand, 2000ULL);
  gen.optimize_envelope(integrand, 5000);

  auto callback = [](const Point<>&, double) {};
  auto res1 = gen.generate_trials(integrand, 5000, callback);
  gen.set_seed(42);
  auto res2 = gen.generate_trials(integrand, 5000, callback);

  GenerationResult<double, unsigned long long> merged;
  merged.accumulate(res1);
  merged.accumulate(res2);

  REQUIRE(merged.n_events() == res1.n_events() + res2.n_events());
  REQUIRE(merged.n_trials() == res1.n_trials() + res2.n_trials());
  REQUIRE(merged.n_nonfinite() == res1.n_nonfinite() + res2.n_nonfinite());
  REQUIRE(merged.volume() == Approx(res1.volume()));
  REQUIRE(merged.value() == Approx(1.5).margin(5.0 * merged.error()));

  // merging results from different envelopes is rejected
  gen.envelope_scale(2.0);
  auto res3 = gen.generate_trials(integrand, 100, callback);
  REQUIRE_THROWS_AS(merged.accumulate(res3), std::invalid_argument);
}
