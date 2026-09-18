#include "kakuhen/integrator/basin_generator.h"
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <span>
#include <sstream>
#include <stdexcept>

using namespace kakuhen::integrator;
using Catch::Approx;

TEST_CASE("BasinGenerator handles correlated integrands with conditional sampling",
          "[basin_generator]") {
  BasinGenerator<> gen(2, 4, 8);
  // strong ridge along x0 == x1 forces a conditional sampling order
  auto integrand = [](const Point<>& point) {
    const double d = point.x[0] - point.x[1];
    return std::exp(-100.0 * d * d);
  };

  gen.integrate(integrand,
                {.neval = 20000, .niter = 5, .adapt = true, .verbosity = 0, .progress_bar = false});
  gen.set_options({.frozen = true, .verbosity = 0});
  auto prod =
      gen.integrate(integrand, {.neval = 20000, .niter = 2, .verbosity = 0, .progress_bar = false});
  // the correlation must be detected, otherwise this test does not exercise
  // the conditional envelope tables
  REQUIRE(gen.nblocks() < 2);

  gen.initialize_envelope(std::abs(prod.value()));
  gen.optimize_envelope(integrand, 20000);

  double weight_sum = 0.0;
  auto callback = [&](const Point<>&, double event_weight) { weight_sum += event_weight; };
  auto result = gen.generate_trials(integrand, 50000, callback);

  REQUIRE(result.status() == GenerationStatus::COMPLETED);
  const double estimate = result.volume() * weight_sum / static_cast<double>(result.n_trials());
  REQUIRE(result.value() == Approx(estimate));
  const double sigma = std::sqrt(result.error() * result.error() + prod.error() * prod.error());
  REQUIRE(result.value() == Approx(prod.value()).margin(5.0 * sigma));
}

namespace {
/// expose the learned sampling order to assert the nesting structure
struct OrderProbeGen : BasinGenerator<> {
  using BasinGenerator<>::BasinGenerator;
  using BasinGenerator<>::order_;
};
}  // namespace

TEST_CASE("BasinGenerator integrates chained conditional envelopes", "[basin_generator]") {
  OrderProbeGen gen(3, 4, 8);
  // x1 rides on x0, x2 rides on x1: forces a conditional->conditional chain
  auto integrand = [](const Point<>& p) {
    const double d1 = p.x[1] - p.x[0];
    const double d2 = p.x[2] - p.x[1];
    return std::exp(-100.0 * d1 * d1 - 100.0 * d2 * d2);
  };

  gen.integrate(integrand,
                {.neval = 30000, .niter = 5, .adapt = true, .verbosity = 0, .progress_bar = false});
  gen.set_options({.frozen = true, .verbosity = 0});
  auto prod =
      gen.integrate(integrand, {.neval = 30000, .niter = 2, .verbosity = 0, .progress_bar = false});

  // both x1 and x2 must be sampled conditionally ...
  REQUIRE(gen.nblocks() == 1);
  // ... and form a chain (one conditional's parent is the other's child), so
  // the envelope exercises refined conditional CDFs
  const auto p1 = gen.order_(1, 0);
  const auto c1 = gen.order_(1, 1);
  const auto p2 = gen.order_(2, 0);
  const auto c2 = gen.order_(2, 1);
  REQUIRE((p2 == c1 || p1 == c2));

  gen.initialize_envelope(std::abs(prod.value()));
  gen.optimize_envelope(integrand, 20000);

  double weight_sum = 0.0;
  auto callback = [&](const Point<>&, double event_weight) { weight_sum += event_weight; };
  auto result = gen.generate_trials(integrand, 50000, callback);

  REQUIRE(result.status() == GenerationStatus::COMPLETED);
  const double estimate = result.volume() * weight_sum / static_cast<double>(result.n_trials());
  REQUIRE(result.value() == Approx(estimate));
  const double sigma = std::sqrt(result.error() * result.error() + prod.error() * prod.error());
  REQUIRE(result.value() == Approx(prod.value()).margin(5.0 * sigma));
}

namespace {
// Exposes the proposal hooks and inverts the map for a proposed point: the
// recorded cells pick the grid cell of each coordinate, and the map is linear
// inside a cell.
struct InvertibleBasin : BasinGenerator<> {
  using BasinGenerator<>::BasinGenerator;
  using BasinGenerator<>::env_propose;
  using BasinGenerator<>::env_value;
  using BasinGenerator<>::make_cell_ctx;

  void u_of(const Point<>& p, const cell_ctx_type& cell, std::span<double> u) const {
    for (S iord = 0; iord < ndim_; ++iord) {
      const S idim = order_(iord, 1);
      double x_low, x_upp;
      S index, ncell;
      if (iord < nblocks_) {
        index = cell(idim, 0);
        ncell = ndiv0_;
        x_low = index > 0 ? ordered_grid0_(iord, index - 1) : 0.0;
        x_upp = ordered_grid0_(iord, index);
      } else {
        const S row = cell(order_(iord, 0), 0) / ndiv2_;
        index = cell(idim, 2);
        ncell = ndiv2_;
        x_low = index > 0 ? ordered_grid_(iord, row, index - 1) : 0.0;
        x_upp = ordered_grid_(iord, row, index);
      }
      const double t = (p.x[idim] - x_low) / (x_upp - x_low);
      REQUIRE(t >= 0.0);
      REQUIRE(t <= 1.0);
      u[idim] = std::min((double(index) + t) / double(ncell), std::nextafter(1.0, 0.0));
    }
  }

  // the fused proposal agrees with map_point at the recovered u (up to rounding)
  void require_mapped(std::span<const double> u, const Point<>& p) const {
    Point<> mapped(ndim_);
    auto cell = make_cell_ctx();
    map_point(u, mapped, cell);
    for (S d = 0; d < ndim_; ++d)
      REQUIRE(mapped.x[d] == Approx(p.x[d]).margin(1e-12));
    REQUIRE(mapped.weight == Approx(p.weight));
  }
};

// A three-coordinate map with prescribed envelope tables. The conditional
// x1 intervals cross x1=1/2, so a chain must account for both reachable rows.
struct PrescribedBasin : InvertibleBasin {
  explicit PrescribedBasin(bool chain) : InvertibleBasin(3, 2, 2), chain_(chain) {
    nblocks_ = 1;
    order_(0, 0) = order_(0, 1) = 0;
    order_(1, 0) = 0;
    order_(1, 1) = 1;
    order_(2, 0) = chain ? 1 : 0;
    order_(2, 1) = 2;
    for (S row = 0; row < 2; ++row) {
      for (S bin = 0; bin < 2; ++bin)
        ordered_grid_(0, row, bin) = double(2 * row + bin + 1) / 4;
      ordered_grid_(1, row, 0) = row == 0 ? 0.375 : 0.625;
      ordered_grid_(1, row, 1) = 1;
      ordered_grid_(2, row, 0) = row == 0 ? 0.25 : 0.75;
      ordered_grid_(2, row, 1) = 1;
    }
    set_options({.frozen = true, .verbosity = 0});
    initialize_envelope(1.0);
    kakuhen::ndarray::NDArray<double, S> raw({3, 2, 2});
    const double values[] = {1, 2, 3, 4, 2, 1, 1, 3, 1, 4, 2, 1};
    std::copy(std::begin(values), std::end(values), raw.begin());
    std::stringstream tables;
    for (S dimension : {S(3), S(2), S(2)})
      kakuhen::util::serialize::serialize_one(tables, dimension);
    raw.serialize(tables);
    read_envelope_table(tables);
    envelope_scale(1.0);
    set_seed(73);
  }

  // Independent raw-product oracle; exact integrals follow from splitting
  // the middle coordinate's two conditional rows at physical x1 = 1/2.
  double proposal_density(const std::array<double, 3>& u, const Point<>& p) const {
    const unsigned root = static_cast<unsigned>(u[0] * 4);
    const unsigned row = root / 2;
    const unsigned j = static_cast<unsigned>(u[1] * 2);
    const unsigned k = static_cast<unsigned>(u[2] * 2);
    const unsigned last_row = chain_ ? unsigned(p.x[1] > 0.5) : row;
    const double a[2][2] = {{2, 1}, {1, 3}};
    const double b[2][2] = {{1, 4}, {2, 1}};
    return double(root + 1) * a[row][j] * b[last_row][k] / (chain_ ? 8.4625 : 8.0625);
  }
  bool chain_;
};
}  // namespace

TEST_CASE("BASIN prescribed branching and chained proposals match their density",
          "[basin_generator]") {
  for (bool chain : {false, true}) {
    CAPTURE(chain);
    PrescribedBasin gen(chain);
    REQUIRE(gen.envelope_volume() == Approx(chain ? 8.4625 : 8.0625));
    std::array<double, 3> u{};
    std::array<unsigned, 16> visits{};
    std::array<double, 16> expected{};
    auto cell = gen.make_cell_ctx();
    Point<> point(3);
    // Every input cell has volume 1/16. For a chain, integrate across the
    // x1=1/2 boundary inside each conditional cell using uniform quadrature.
    constexpr unsigned subdivisions = 120;
    for (unsigned root = 0; root < 4; ++root)
      for (unsigned j = 0; j < 2; ++j)
        for (unsigned k = 0; k < 2; ++k)
          for (unsigned sub = 0; sub < subdivisions; ++sub) {
            u = {(root + 0.5) / 4, (j + (sub + 0.5) / subdivisions) / 2, (k + 0.5) / 2};
            gen.map_point(u, point, cell);
            expected[root * 4 + j * 2 + k] += gen.proposal_density(u, point) / (16 * subdivisions);
          }
    constexpr unsigned samples = 30000;
    for (unsigned sample = 0; sample < samples; ++sample) {
      const double bound = gen.env_propose(point, cell);
      REQUIRE(bound == gen.env_value(cell));
      gen.u_of(point, cell, u);
      gen.require_mapped(u, point);
      const double q = gen.proposal_density(u, point);
      REQUIRE(bound / gen.envelope_volume() == Approx(q));
      ++visits[unsigned(u[0] * 4) * 4 + unsigned(u[1] * 2) * 2 + unsigned(u[2] * 2)];
    }
    for (unsigned bin = 0; bin < visits.size(); ++bin) {
      const double p = expected[bin];
      REQUIRE(double(visits[bin]) / samples ==
              Approx(p).margin(5 * std::sqrt(p * (1 - p) / samples)));
    }
  }
}

TEST_CASE("BASIN prescribed maps preserve weighted physical moments", "[basin_generator]") {
  for (bool chain : {false, true}) {
    CAPTURE(chain);
    PrescribedBasin gen(chain);
    // Deliberately undersize the envelope to exercise overweight correction.
    gen.envelope_scale(0.05);
    std::array<IntegralAccumulator<double, unsigned long long>, 4> moments;
    const auto result =
        gen.generate_trials([](const Point<>& p) { return 8 * p.x[0] * p.x[1] * p.x[2]; }, 40000,
                            [&](const Point<>& p, double weight) {
                              for (unsigned d = 0; d < 3; ++d)
                                moments[d].accumulate(weight * p.x[d]);
                              moments[3].accumulate(weight * p.x[0] * p.x[2]);
                            });
    REQUIRE(result.status() == GenerationStatus::COMPLETED);
    REQUIRE(result.n_overweight() > 0);
    REQUIRE(result.value() == Approx(1).margin(5 * result.error()));
    for (unsigned d = 0; d < moments.size(); ++d) {
      moments[d].accumulate_zeros(result.n_trials() - moments[d].count());
      const double target = d < 3 ? 2.0 / 3 : 4.0 / 9;
      REQUIRE(result.volume() * moments[d].value() ==
              Approx(target).margin(5 * result.volume() * moments[d].error()));
    }
  }
}

namespace {
// Two roots, a four-node path, and a branch at an internal node. Physical
// dimensions deliberately differ from sampling-order indices.
struct ForestBasin : InvertibleBasin {
  ForestBasin() : InvertibleBasin(6, 2, 2) {
    nblocks_ = 2;
    const S parents[] = {1, 4, 4, 2, 2, 0};
    const S children[] = {1, 4, 2, 0, 3, 5};
    kakuhen::ndarray::NDArray<double, S> raw({6, 2, 2});
    for (S j = 0; j < 6; ++j) {
      order_(j, 0) = parents[j];
      order_(j, 1) = children[j];
      for (S c = 0; c < 2; ++c)
        for (S k = 0; k < 2; ++k) {
          ordered_grid_(j, c, k) =
              j < 2 ? double(2 * c + k + 1) / 4 : (k == 1 ? 1.0 : (c == 0 ? 0.375 : 0.625));
          grid_(parents[j], children[j], c, k) = ordered_grid_(j, c, k);
          raw(j, c, k) = 1 + (j + 2 * c + k) % 5;
        }
    }
    set_options({.frozen = true, .verbosity = 0});
    initialize_envelope(1.0);
    std::stringstream tables;
    for (S size : {S(6), S(2), S(2)})
      kakuhen::util::serialize::serialize_one(tables, size);
    raw.serialize(tables);
    read_envelope_table(tables);
    envelope_scale(1.0);
    set_seed(149);
  }
  using BasinGenerator<>::uniform_distribution_;
};
}  // namespace

TEST_CASE("BASIN exact sampler integrates a deep forest", "[basin_generator]") {
  ForestBasin gen;
  std::array<double, 6> u{};
  Point<> point(6);
  auto cell = gen.make_cell_ctx();
  double mass = 0;
  std::array<double, 6> first_moments{};
  // Every conditional split lies at u=0.4 or 0.6, and raw cell boundaries
  // lie at u=0.5. Tensor midpoint quadrature is therefore exact for R and uR.
  constexpr unsigned bins[] = {10, 4, 10, 10, 4, 10};
  constexpr unsigned count = 160000;
  for (unsigned sample = 0; sample < count; ++sample) {
    unsigned index = sample;
    for (unsigned d = 0; d < 6; ++d) {
      u[d] = (index % bins[d] + 0.5) / bins[d];
      index /= bins[d];
    }
    gen.map_point(u, point, cell);
    const double raw = gen.env_value(cell);
    mass += raw;
    for (unsigned d = 0; d < 6; ++d)
      first_moments[d] += raw * u[d];
  }
  REQUIRE(gen.envelope_volume() == Approx(mass / count));
  std::array<IntegralAccumulator<double, unsigned long long>, 6> observed;
  for (unsigned sample = 0; sample < 30000; ++sample) {
    const double raw = gen.env_propose(point, cell);
    REQUIRE(raw == gen.env_value(cell));
    gen.u_of(point, cell, u);
    gen.require_mapped(u, point);
    for (unsigned d = 0; d < 6; ++d)
      observed[d].accumulate(u[d]);
  }
  for (unsigned d = 0; d < 6; ++d)
    REQUIRE(observed[d].value() == Approx(first_moments[d] / mass).margin(5 * observed[d].error()));

  SECTION("scaling preserves proposals and scales the raw product") {
    ForestBasin scaled;
    gen.set_seed(211);
    scaled.set_seed(211);
    scaled.envelope_scale(64.0);  // exactly doubles every height
    REQUIRE(scaled.envelope_volume() == Approx(64 * gen.envelope_volume()));
    Point<> other(6);
    for (unsigned sample = 0; sample < 100; ++sample) {
      REQUIRE(scaled.env_propose(other) == 64 * gen.env_propose(point));
      REQUIRE(other.x == point.x);
      REQUIRE(other.weight == point.weight);
    }
  }

  SECTION("recording the cells does not change the proposal") {
    ForestBasin twin;
    gen.set_seed(97);
    twin.set_seed(97);
    Point<> other(6);
    for (unsigned sample = 0; sample < 100; ++sample) {
      REQUIRE(twin.env_propose(other) == gen.env_propose(point, cell));
      REQUIRE(other.x == point.x);
      REQUIRE(other.weight == point.weight);
    }
  }

  SECTION("CDF endpoints stay inside the drawn cells") {
    for (double edge : {0.0, 0.5, 1.0}) {
      gen.uniform_distribution_ = std::uniform_real_distribution<double>(
          edge, std::nextafter(edge, std::numeric_limits<double>::infinity()));
      const double raw = gen.env_propose(point, cell);
      REQUIRE(raw == gen.env_value(cell));
      for (double coordinate : point.x) {
        REQUIRE(coordinate >= 0);
        REQUIRE(coordinate <= 1);
      }
      gen.u_of(point, cell, u);  // checks that each x lies in its recorded cell
      gen.require_mapped(u, point);
    }
  }

  SECTION("loading rebuilds the exact CDFs") {
    std::stringstream state;
    gen.write_state_stream(state);
    ForestBasin restored;
    restored.read_state_stream(state);
    REQUIRE(restored.envelope_volume() == gen.envelope_volume());
    gen.set_seed(311);
    restored.set_seed(311);
    Point<> other(6);
    for (unsigned sample = 0; sample < 100; ++sample) {
      REQUIRE(restored.env_propose(other) == gen.env_propose(point));
      REQUIRE(other.x == point.x);
      REQUIRE(other.weight == point.weight);
    }
  }
}
