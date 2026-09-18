#include "kakuhen/integrator/basin.h"
#include "kakuhen/integrator/plain.h"
#include "kakuhen/integrator/vegas.h"
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <span>
#include <vector>

using namespace kakuhen::integrator;
using Catch::Approx;

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
