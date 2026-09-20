#include "kakuhen/integrator/plain.h"
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace kakuhen::integrator;

TEST_CASE("Progress EventSignal bit operations", "[integrator][progress]") {
  EventSignal s = EventSignal::NONE;
  s |= EventSignal::STOP;
  s |= EventSignal::EXCEPTION;
  REQUIRE(has_signal(s, EventSignal::STOP));
  REQUIRE(has_signal(s, EventSignal::EXCEPTION));
  REQUIRE(!has_signal(s, EventSignal::NONE));
}

TEST_CASE("Progress callback receives milestones and lifecycle events", "[integrator][progress]") {
  Plain<> plain(2);
  using opts_t = Plain<>::options_type;
  using event_t = Plain<>::progress_event_type;

  int start_count = 0;
  int end_count = 0;
  int iter_start_count = 0;
  int iter_end_count = 0;
  int eval_milestone_count = 0;

  // progress_step = 0.5 on neval=10 -> step=5 -> 2 milestones/iter * 2 iters = 4
  opts_t opts{
      .neval = 10,
      .niter = 2,
      .verbosity = 0,
      .progress_step = 0.5,
  };

  auto callback = [&](const event_t& ev) {
    if (ev.kind == ProgressEventKind::START) {
      start_count++;
      REQUIRE(ev.current_eval == 0);
      REQUIRE(ev.fraction == 0.0);
      REQUIRE(ev.niter == 2);
    }
    if (ev.kind == ProgressEventKind::ITER_START) {
      iter_start_count++;
      REQUIRE(ev.current_iter < ev.niter);
      REQUIRE(ev.current_eval == 0);
    }
    if (ev.kind == ProgressEventKind::ITER_END) {
      iter_end_count++;
      REQUIRE(ev.current_iter < ev.niter);
      REQUIRE(ev.fraction == 1.0);
      REQUIRE(ev.current_eval == ev.neval);
    }
    if (ev.kind == ProgressEventKind::EVAL_MILESTONE) {
      eval_milestone_count++;
      REQUIRE(ev.current_eval <= ev.neval);
      REQUIRE(ev.fraction > 0.0);
      REQUIRE(ev.fraction <= 1.0);
    }
    if (ev.kind == ProgressEventKind::END) {
      end_count++;
      REQUIRE(ev.fraction == 1.0);
      REQUIRE(ev.current_eval == ev.neval);
    }
    return EventSignal::NONE;
  };

  auto res = plain.integrate([](const Point<>&) { return 1.0; }, opts, callback);
  REQUIRE(res.count() == 20);
  REQUIRE(start_count == 1);
  REQUIRE(iter_start_count == 2);
  REQUIRE(iter_end_count == 2);
  REQUIRE(eval_milestone_count == 4);
  REQUIRE(end_count == 1);
}

TEST_CASE("Progress callback can stop integration", "[integrator][progress]") {
  Plain<> plain(2);
  using opts_t = Plain<>::options_type;
  using event_t = Plain<>::progress_event_type;

  opts_t opts{
      .neval = 100,
      .niter = 1,
      .verbosity = 0,
      .progress_step = 0.05,
  };

  auto callback = [](const event_t& ev) {
    if (ev.kind == ProgressEventKind::EVAL_MILESTONE && ev.current_eval >= 15) {
      return EventSignal::STOP;
    }
    return EventSignal::NONE;
  };

  auto res = plain.integrate([](const Point<>&) { return 1.0; }, opts, callback);
  REQUIRE(res.count() > 0);
  REQUIRE(res.count() < 100);
}

TEST_CASE("Progress callback exceptions are converted to EXCEPTION|STOP",
          "[integrator][progress]") {
  Plain<> plain(2);
  using opts_t = Plain<>::options_type;
  using event_t = Plain<>::progress_event_type;

  std::vector<ProgressEventKind> seen;
  opts_t opts{
      .neval = 100,
      .niter = 1,
      .verbosity = 0,
      .progress_step = 0.1,
  };

  auto callback = [&](const event_t& ev) {
    seen.push_back(ev.kind);
    if (ev.kind == ProgressEventKind::START) {
      throw std::runtime_error("boom");
    }
    return EventSignal::NONE;
  };

  auto res = plain.integrate([](const Point<>&) { return 1.0; }, opts, callback);
  REQUIRE(res.count() == 0);
  REQUIRE(!seen.empty());
  REQUIRE(seen.front() == ProgressEventKind::START);
  REQUIRE(seen.back() == ProgressEventKind::END);
}

TEST_CASE("Invalid progress step is rejected", "[integrator][progress]") {
  Plain<> plain(2);
  using opts_t = Plain<>::options_type;
  using event_t = Plain<>::progress_event_type;

  auto callback = [](const event_t&) { return EventSignal::NONE; };

  for (const double step : {0.0, -0.5, 1.0 + 1e-12, std::numeric_limits<double>::quiet_NaN()}) {
    const opts_t opts{.neval = 10, .niter = 1, .verbosity = 0, .progress_step = step};
    REQUIRE_THROWS_AS(plain.integrate([](const Point<>&) { return 1.0; }, opts, callback),
                      std::invalid_argument);
  }
}

TEST_CASE("Invalid integration options are rejected before touching any state",
          "[integrator][progress]") {
  using event_t = Plain<>::progress_event_type;
  auto integrand = [](const Point<>& p) { return p.x[0]; };
  auto callback = [](const event_t&) { return EventSignal::NONE; };

  Plain<> plain(2);
  plain.set_options({.verbosity = 0});
  REQUIRE_THROWS_AS(plain.integrate(integrand, {.neval = 0, .niter = 1, .seed = 42}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(plain.integrate(integrand, {.neval = 10, .niter = 0, .seed = 42}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(plain.integrate(integrand,
                                    {.neval = 10,
                                     .niter = 1,
                                     .seed = 42,
                                     .progress_step = std::numeric_limits<double>::quiet_NaN()},
                                    callback),
                    std::invalid_argument);

  // neither the options nor the RNG were touched: same stream as a fresh integrator
  Plain<> fresh(2);
  fresh.set_options({.verbosity = 0});
  REQUIRE(plain.seed() == fresh.seed());
  const auto res = plain.integrate(integrand, {.neval = 100, .niter = 1});
  const auto res_fresh = fresh.integrate(integrand, {.neval = 100, .niter = 1});
  REQUIRE(res.value() == res_fresh.value());
}

TEST_CASE("Progress event includes value, error, and elapsed time", "[integrator][progress]") {
  Plain<> plain(2);
  using opts_t = Plain<>::options_type;
  using event_t = Plain<>::progress_event_type;

  opts_t opts{
      .neval = 100,
      .niter = 2,
      .verbosity = 0,
      .progress_step = 0.5,
  };

  bool checked_value = false;
  auto callback = [&](const event_t& ev) {
    // After first iteration ends, we should have accumulated results
    if (ev.kind == ProgressEventKind::ITER_END && ev.current_iter == 0) {
      REQUIRE(ev.value > 0.0);
      REQUIRE(ev.error >= 0.0);
      REQUIRE(ev.elapsed_start > 0.0);
      REQUIRE(ev.elapsed_iter > 0.0);
      checked_value = true;
    }
    return EventSignal::NONE;
  };

  auto res = plain.integrate([](const Point<>&) { return 1.0; }, opts, callback);
  REQUIRE(checked_value);
}

TEST_CASE("Integration without callback works normally", "[integrator][progress]") {
  Plain<> plain(2);
  using opts_t = Plain<>::options_type;

  opts_t opts{
      .neval = 100,
      .niter = 2,
      .verbosity = 0,
  };

  // Call without callback - should compile and work
  auto res = plain.integrate([](const Point<>&) { return 1.0; }, opts);
  REQUIRE(res.count() == 200);
}
