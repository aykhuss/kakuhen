#include "kakuhen/integrator/basin.h"
#include "kakuhen/integrator/plain.h"
#include "kakuhen/integrator/vegas.h"
#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <span>
#include <vector>

using namespace kakuhen::integrator;
using Catch::Approx;

namespace {
// isolated root 1 and the tree 4 -> 2 -> 0 -> 5, with a branch 2 -> 3
struct ForestBasin : Basin<> {
  explicit ForestBasin(bool forest = true) : Basin<>(6, 2, 2) {
    if (!forest) return;
    nblocks_ = 2;
    const S parents[] = {1, 4, 4, 2, 2, 0};
    const S children[] = {1, 4, 2, 0, 3, 5};
    for (S i = 0; i < 6; ++i) {
      order_(i, 0) = parents[i];
      order_(i, 1) = children[i];
      for (S row = 0; row < 2; ++row)
        for (S bin = 0; bin < 2; ++bin) {
          const double edge =
              i < 2 ? double(2 * row + bin + 1) / 4 : (bin == 1 ? 1.0 : (row == 0 ? 0.5 : 0.75));
          grid_(parents[i], children[i], row, bin) = ordered_grid_(i, row, bin) = edge;
        }
    }
  }
};
}  // namespace

TEST_CASE("Plain map_point maps supplied uniforms", "[map_point][plain]") {
  Plain<> plain(3);
  Point<> point(3);
  point.sample_index = 42;

  const std::array<double, 3> u{0.125, 0.5, 0.875};

  plain.map_point(std::span<const double>{u.data(), u.size()}, point);

  REQUIRE(point.x[0] == Approx(u[0]));
  REQUIRE(point.x[1] == Approx(u[1]));
  REQUIRE(point.x[2] == Approx(u[2]));
  REQUIRE(point.weight == Approx(1.0));
  REQUIRE(point.sample_index == 42);
}

TEST_CASE("Vegas map_point maps supplied uniforms and cells", "[map_point][vegas]") {
  Vegas<> vegas(2, 4);
  Point<> point(2);
  point.sample_index = 7;

  const std::array<double, 2> u{0.125, 0.875};
  Vegas<>::cell_ctx_type cell({2});

  vegas.map_point(std::span<const double>{u.data(), u.size()}, point, cell);

  REQUIRE(point.x[0] == Approx(u[0]));
  REQUIRE(point.x[1] == Approx(u[1]));
  REQUIRE(point.weight == Approx(1.0));
  REQUIRE(point.sample_index == 7);
  REQUIRE(cell[0] == 0);
  REQUIRE(cell[1] == 3);
}

TEST_CASE("Basin map_point maps supplied uniforms and diagonal cells", "[map_point][basin]") {
  Basin<> basin(2, 2, 3);
  Point<> point(2);
  point.sample_index = 11;

  const std::array<double, 2> u{0.2, 0.83};
  Basin<>::cell_ctx_type cell({2, 3});

  basin.map_point(std::span<const double>{u.data(), u.size()}, point, cell);

  REQUIRE(point.x[0] == Approx(u[0]));
  REQUIRE(point.x[1] == Approx(u[1]));
  REQUIRE(point.weight == Approx(1.0));
  REQUIRE(point.sample_index == 11);
  REQUIRE(cell(0, 0) == static_cast<Basin<>::size_type>(u[0] * basin.ndiv0()));
  REQUIRE(cell(1, 0) == static_cast<Basin<>::size_type>(u[1] * basin.ndiv0()));
  // fresh grid: every dimension is sampled diagonally
  REQUIRE(cell(0, 1) == 0);
  REQUIRE(cell(1, 1) == 1);
  REQUIRE(cell(0, 2) == basin.ndiv2());
  REQUIRE(cell(1, 2) == basin.ndiv2());
}

TEST_CASE("Basin lazy cells preserve mapping across forest boundaries", "[map_point][basin]") {
  for (bool forest : {false, true}) {
    ForestBasin basin(forest);
    Point<> full(6), lazy(6);
    full.sample_index = lazy.sample_index = 42;
    Basin<>::cell_ctx_type full_cell({6, 3}), lazy_cell({6, 3});
    lazy_cell.fill(99);
    std::array<double, 6> u{};
    const std::array<double, 4> edges{0.25, 0.5, 0.75, 1.0};
    const std::array<Basin<>::S, 6> parents{2, 1, 4, 2, 4, 0};
    for (double value : {0.0, 0.75, 0.25, std::nextafter(0.5, 0.0), 0.5, std::nextafter(0.5, 1.0),
                         std::nextafter(1.0, 0.0), 0.0}) {
      CAPTURE(forest, value);
      u.fill(value);
      u[4] = 0.25;  // x2 = u2; its children change rows just above x2 = 1/2
      basin.map_point(u, full, full_cell);
      basin.map_point_impl(u, lazy, lazy_cell, false);
      const bool upper = forest && value > 0.5;
      const double child = upper ? 0.5 + 0.5 * value : value;
      const std::array<double, 6> expected{child, value, value, child, 0.25, child};
      REQUIRE(full.weight == (upper ? 0.125 : 1.0));
      REQUIRE(lazy.weight == full.weight);
      REQUIRE(lazy.x == full.x);
      REQUIRE(full.sample_index == 42);
      REQUIRE(lazy.sample_index == 42);
      for (Basin<>::S d = 0; d < 6; ++d) {
        CAPTURE(d);
        REQUIRE(full.x[d] == Approx(expected[d]).margin(1e-15));
        const bool root = !forest || d == 1 || d == 4;
        const auto diagonal = root ? static_cast<Basin<>::S>(4 * u[d])
                                   : std::count_if(edges.begin(), edges.end(),
                                                   [&](double edge) { return edge < full.x[d]; });
        REQUIRE(full_cell(d, 0) == diagonal);
        REQUIRE(lazy_cell(d, 0) == (forest && (d == 3 || d == 5) ? 4 : diagonal));
        REQUIRE(full_cell(d, 1) == (root ? d : parents[d]));
        REQUIRE(full_cell(d, 2) == (root ? 2 : static_cast<Basin<>::S>(2 * u[d])));
        REQUIRE(lazy_cell(d, 1) == full_cell(d, 1));
        REQUIRE(lazy_cell(d, 2) == full_cell(d, 2));
      }
      if (value <= 0.5) {
        // reuse both partial and complete contexts on subsequent samples
        basin.map_point(u, lazy, lazy_cell);
        REQUIRE(std::equal(full_cell.begin(), full_cell.end(), lazy_cell.begin()));
      }
    }
  }
}

TEST_CASE("Basin frozen and collecting runs sample the same forest", "[map_point][basin]") {
  ForestBasin basin;
  std::vector<Point<>> samples;
  basin.set_seed(73);
  const auto collected = basin.integrate(
      [&](const Point<>& p) {
        samples.push_back(p);
        return 1.0 + p.x[0];
      },
      {.neval = 128, .niter = 1, .adapt = false, .verbosity = 0});
  basin.set_seed(73);
  const auto frozen = basin.integrate(
      [&](const Point<>& p) {
        REQUIRE(p.x == samples[p.sample_index].x);
        REQUIRE(p.weight == samples[p.sample_index].weight);
        return 1.0 + p.x[0];
      },
      {.neval = 128, .niter = 1, .frozen = true, .verbosity = 0});
  REQUIRE(frozen.value() == collected.value());
  REQUIRE(frozen.error() == collected.error());
}

TEST_CASE("Adaptive map_point is deterministic for fixed input", "[map_point]") {
  auto integrand = [](const Point<>& point) {
    const double dx = point.x[0] - point.x[1];
    return 1.0 / (0.01 + dx * dx);
  };

  SECTION("Vegas") {
    Vegas<> vegas(2, 8);
    vegas.integrate(integrand, {.neval = 512, .niter = 2, .adapt = true, .verbosity = 0});

    const std::array<double, 2> u{0.13, 0.77};
    Point<> point_a(2);
    Point<> point_b(2);
    Vegas<>::cell_ctx_type cell_a({2});
    Vegas<>::cell_ctx_type cell_b({2});

    vegas.map_point(std::span<const double>{u.data(), u.size()}, point_a, cell_a);
    vegas.map_point(std::span<const double>{u.data(), u.size()}, point_b, cell_b);

    REQUIRE(point_a.x[0] == Approx(point_b.x[0]));
    REQUIRE(point_a.x[1] == Approx(point_b.x[1]));
    REQUIRE(point_a.weight == Approx(point_b.weight));
    REQUIRE(cell_a[0] == cell_b[0]);
    REQUIRE(cell_a[1] == cell_b[1]);
    REQUIRE(cell_a[0] < vegas.ndiv());
    REQUIRE(cell_a[1] < vegas.ndiv());
  }

  SECTION("Basin") {
    Basin<> basin(2, 2, 4);
    basin.set_min_score(0.0);
    basin.integrate(integrand, {.neval = 512, .niter = 2, .adapt = true, .verbosity = 0});

    const std::array<double, 2> u{0.13, 0.77};
    Point<> point_a(2);
    Point<> point_b(2);
    Basin<>::cell_ctx_type cell_a({2, 3});
    Basin<>::cell_ctx_type cell_b({2, 3});

    basin.map_point(std::span<const double>{u.data(), u.size()}, point_a, cell_a);
    basin.map_point(std::span<const double>{u.data(), u.size()}, point_b, cell_b);

    REQUIRE(point_a.x[0] == Approx(point_b.x[0]));
    REQUIRE(point_a.x[1] == Approx(point_b.x[1]));
    REQUIRE(point_a.weight == Approx(point_b.weight));
    for (Basin<>::size_type idim = 0; idim < 2; ++idim) {
      REQUIRE(cell_a(idim, 0) == cell_b(idim, 0));
      REQUIRE(cell_a(idim, 1) == cell_b(idim, 1));
      REQUIRE(cell_a(idim, 2) == cell_b(idim, 2));
      REQUIRE(cell_a(idim, 0) < basin.ndiv0());
      // diagonal dimensions carry the `ndiv2` sentinel; conditional ones a
      // valid sub-grid cell of the conditioning dimension
      if (cell_a(idim, 1) == idim) {
        REQUIRE(cell_a(idim, 2) == basin.ndiv2());
      } else {
        REQUIRE(cell_a(idim, 2) < basin.ndiv2());
      }
    }
  }
}
