#include "kakuhen/integrator/basin.h"
#include "kakuhen/integrator/vegas.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using kakuhen::integrator::Basin;
using kakuhen::integrator::Point;
using kakuhen::integrator::Vegas;
using Clock = std::chrono::steady_clock;
using Count = Vegas<>::count_type;
using Size = Vegas<>::size_type;

struct Config {
  std::string case_name = "all";
  std::string algorithm = "both";
  std::string mode = "frozen";
  Count repeats = 12;
  Count seed = 101;
  Count warmup_evals = 50000;
  Count warmup_iters = 6;
  Count evals = 100000;
  Count iters = 4;
  Size bins = 128;
  Size coarse = 8;
  Size fine = 16;
  Size dimensions = 8;
};

double gaussian_integral(double alpha, double center) {
  const double root = std::sqrt(alpha);
  return std::sqrt(std::numbers::pi / alpha) *
         (std::erf(root * center) + std::erf(root * (1.0 - center))) / 2.0;
}

struct Constant {
  double operator()(const Point<>&) const {
    return 1.0;
  }
};

struct Smooth {
  double operator()(const Point<>& point) const {
    double value = 1.0;
    for (const double x : point.x)
      value *= 0.5 + x;
    return value;
  }
};

struct Gaussian {
  std::array<double, 8> centers{};
  double normalization = 1.0;

  Gaussian() {
    for (std::size_t d = 0; d < centers.size(); ++d) {
      centers[d] = 0.35 + 0.03 * static_cast<double>(d);
      normalization *= gaussian_integral(40.0, centers[d]);
    }
  }

  double operator()(const Point<>& point) const {
    double distance = 0.0;
    for (std::size_t d = 0; d < centers.size(); ++d) {
      const double delta = point.x[d] - centers[d];
      distance += delta * delta;
    }
    return std::exp(-40.0 * distance) / normalization;
  }
};

struct Ridge {
  double normalization;

  Ridge() {
    const double pair_integral =
        std::sqrt(std::numbers::pi / 400.0) * std::erf(20.0) - (1.0 - std::exp(-400.0)) / 400.0;
    normalization = pair_integral * pair_integral;
  }

  double operator()(const Point<>& point) const {
    const double a = point.x[0] - point.x[1];
    const double b = point.x[2] - point.x[3];
    return std::exp(-400.0 * (a * a + b * b)) / normalization;
  }
};

struct Multimodal {
  double normalization = 0.0;

  Multimodal() {
    for (const double center : {0.25, 0.5, 0.75}) {
      normalization += std::pow(gaussian_integral(60.0, center), 8);
    }
  }

  double operator()(const Point<>& point) const {
    double value = 0.0;
    for (const double center : {0.25, 0.5, 0.75}) {
      double distance = 0.0;
      for (const double x : point.x) {
        const double delta = x - center;
        distance += delta * delta;
      }
      value += std::exp(-60.0 * distance);
    }
    return value / normalization;
  }
};

struct Discontinuous {
  double operator()(const Point<>& point) const {
    return point.x[0] + point.x[1] < 1.0 && point.x[2] + point.x[3] < 1.0 ? 4.0 : 0.0;
  }
};

// An analytically normalized sum with 64 exponentials per evaluation. This
// tests whether sampler overhead matters when evaluating the integrand costs more.
struct Expensive {
  std::array<std::array<double, 8>, 64> centers{};
  double normalization = 0.0;

  Expensive() {
    for (std::size_t k = 0; k < centers.size(); ++k) {
      double integral = 1.0;
      for (std::size_t d = 0; d < centers[k].size(); ++d) {
        centers[k][d] = 0.2 + 0.6 * static_cast<double>((k * (2 * d + 1) + 7 * d) % 67) / 66.0;
        integral *= gaussian_integral(15.0, centers[k][d]);
      }
      normalization += integral;
    }
  }

  double operator()(const Point<>& point) const {
    double value = 0.0;
    for (const auto& center : centers) {
      double distance = 0.0;
      for (std::size_t d = 0; d < center.size(); ++d) {
        const double delta = point.x[d] - center[d];
        distance += delta * delta;
      }
      value += std::exp(-15.0 * distance);
    }
    return value / normalization;
  }
};

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename Integrator, typename Integrand>
void measure(Integrator& integrator, Integrand& integrand, const Config& config,
             const std::string& case_name, const char* algorithm, Count repeat, double setup_s) {
  const Count seed = config.seed + repeat;
  integrator.set_seed(seed);
  integrator.set_options({.verbosity = 0, .progress_bar = false});
  double warm_sample_s = 0.0;
  double warm_adapt_s = 0.0;
  for (Count i = 0; i < config.warmup_iters; ++i) {
    auto start = Clock::now();
    const auto result = integrator.integrate(
        integrand, {.neval = config.warmup_evals, .niter = 1, .adapt = false, .frozen = false});
    warm_sample_s += seconds_since(start);
    if (result.count() != config.warmup_evals) throw std::runtime_error("Warmup count mismatch");
    start = Clock::now();
    integrator.adapt();
    warm_adapt_s += seconds_since(start);
  }

  typename Integrator::result_type combined;
  double sample_s = 0.0;
  double adapt_s = 0.0;
  for (Count i = 0; i < config.iters; ++i) {
    auto start = Clock::now();
    const auto result = integrator.integrate(
        integrand,
        {.neval = config.evals, .niter = 1, .adapt = false, .frozen = config.mode == "frozen"});
    sample_s += seconds_since(start);
    combined.accumulate(result);
    if (config.mode == "adaptive") {
      start = Clock::now();
      integrator.adapt();
      adapt_s += seconds_since(start);
    }
  }
  if (combined.count() != config.evals * config.iters) {
    throw std::runtime_error("Production count mismatch");
  }
  const double value = combined.value();
  const double error = combined.error();
  // On a frozen grid these entries sample the same distribution. Pooling their
  // sums provides a check on inverse-variance weighting for rare large weights.
  typename Integrator::int_acc_type pooled;
  for (const auto& entry : combined.entries())
    pooled.accumulate(entry);
  if (!std::isfinite(value) || !std::isfinite(error)) {
    throw std::runtime_error("Non-finite benchmark result");
  }
  Size blocks = integrator.ndim();
  if constexpr (requires { integrator.nblocks(); }) blocks = integrator.nblocks();
  std::cout << case_name << ',' << algorithm << ',' << config.mode << ',' << integrator.ndim()
            << ',' << config.bins << ',' << config.coarse << ',' << config.fine << ',' << seed
            << ',' << config.warmup_evals << ',' << config.warmup_iters << ',' << config.evals
            << ',' << config.iters << ',' << setup_s << ',' << warm_sample_s << ',' << warm_adapt_s
            << ',' << sample_s << ',' << adapt_s << ',' << value << ',' << error << ','
            << 1.0  // exact: every case is normalized to one
            << ',' << blocks << ',' << pooled.value() << ',' << pooled.error() << '\n';
  std::cout.flush();
}

template <typename Integrand>
void run_case(const Config& config, const std::string& name, Size dimensions, Integrand integrand,
              Count repeat) {
  if (config.case_name != "all" && config.case_name != name) return;
  const auto run = [&](const char* algorithm, auto make_integrator) {
    if (config.algorithm != "both" && config.algorithm != algorithm) return;
    const auto start = Clock::now();
    auto integrator = make_integrator();
    const double setup_s = seconds_since(start);
    measure(integrator, integrand, config, name, algorithm, repeat, setup_s);
  };
  const auto vegas = [&] { run("vegas", [&] { return Vegas<>(dimensions, config.bins); }); };
  const auto basin = [&] {
    run("basin", [&] { return Basin<>(dimensions, config.coarse, config.fine); });
  };
  // Alternate paired order to reduce systematic clock/temperature/order effects.
  if (repeat % 2 == 0) {
    vegas();
    basin();
  } else {
    basin();
    vegas();
  }
}

Config parse_args(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--help") {
      std::cout << "Options (each takes one value):\n"
                   "  --case all|constant|smooth_8d|gaussian_8d|ridge_4d|multimodal_8d|"
                   "discontinuous_4d|expensive_8d (all excludes constant)\n"
                   "  --algorithm both|vegas|basin  --mode frozen|collect|adaptive\n"
                   "  --repeats 12  --seed 101  --warmup-evals 50000  --warmup-iters 6\n"
                   "  --evals 100000  --iters 4  --bins 128  --coarse 8  --fine 16\n"
                   "  --dimensions 8 (constant case only)\n";
      std::exit(0);
    }
    if (++i == argc) throw std::invalid_argument("Missing value for " + key);
    const std::string value(argv[i]);
    if (key == "--case")
      config.case_name = value;
    else if (key == "--algorithm")
      config.algorithm = value;
    else if (key == "--mode")
      config.mode = value;
    else {
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Expected non-negative integer for " + key);
      }
      const auto number = std::stoull(value);
      const auto as_size = [&] {
        if (!std::in_range<Size>(number)) {
          throw std::invalid_argument("Value out of range for " + key);
        }
        return static_cast<Size>(number);
      };
      if (key == "--repeats")
        config.repeats = number;
      else if (key == "--seed")
        config.seed = number;
      else if (key == "--warmup-evals")
        config.warmup_evals = number;
      else if (key == "--warmup-iters")
        config.warmup_iters = number;
      else if (key == "--evals")
        config.evals = number;
      else if (key == "--iters")
        config.iters = number;
      else if (key == "--bins")
        config.bins = as_size();
      else if (key == "--coarse")
        config.coarse = as_size();
      else if (key == "--fine")
        config.fine = as_size();
      else if (key == "--dimensions")
        config.dimensions = as_size();
      else
        throw std::invalid_argument("Unknown option " + key);
    }
  }
  const std::vector<std::string> names{
      "all",      "constant",      "smooth_8d",        "gaussian_8d",
      "ridge_4d", "multimodal_8d", "discontinuous_4d", "expensive_8d"};
  if (std::ranges::find(names, config.case_name) == names.end() ||
      (config.algorithm != "both" && config.algorithm != "vegas" && config.algorithm != "basin") ||
      (config.mode != "frozen" && config.mode != "collect" && config.mode != "adaptive") ||
      config.repeats == 0 || config.iters == 0 || config.evals < 2 || config.warmup_evals < 2 ||
      config.bins < 2 || config.coarse < 2 || config.fine < 2 || config.dimensions == 0) {
    throw std::invalid_argument("Invalid benchmark configuration; use --help");
  }
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parse_args(argc, argv);
    std::cout
        << std::setprecision(17)
        << "case,algorithm,mode,dimensions,bins,coarse,fine,seed,warmup_evals,warmup_iters,"
           "evals,iters,setup_s,warm_sample_s,warm_adapt_s,sample_s,adapt_s,value,error,exact,"
           "blocks,pooled_value,pooled_error\n";
    for (Count repeat = 0; repeat < config.repeats; ++repeat) {
      if (config.case_name == "constant") {
        run_case(config, "constant", config.dimensions, Constant{}, repeat);
      } else {
        run_case(config, "smooth_8d", 8, Smooth{}, repeat);
        run_case(config, "gaussian_8d", 8, Gaussian{}, repeat);
        run_case(config, "ridge_4d", 4, Ridge{}, repeat);
        run_case(config, "multimodal_8d", 8, Multimodal{}, repeat);
        run_case(config, "discontinuous_4d", 4, Discontinuous{}, repeat);
        run_case(config, "expensive_8d", 8, Expensive{}, repeat);
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
