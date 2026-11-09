#include "kakuhen/integrator/basin.h"
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace kakuhen::integrator;
using Catch::Approx;

static auto test_integrand = [](const Point<>& point) {
  const auto& x = point.x;  // shorthand
  return (x[0] + x[1]) / (1. + x[0] - x[1]);
};

TEST_CASE("Basin ndim", "[basin]") {
  auto bas = Basin(3);
  REQUIRE(bas.ndim() == 3);
}

TEST_CASE("write/load state and data", "[basin]") {
  std::stringstream ss;

  auto veg = Basin(2);
  veg.set_options({.verbosity = 0});

  /// quick adaption:  save state
  veg.integrate(test_integrand, {.neval = 1000, .niter = 10, .adapt = true});
  veg.write_state_stream(ss);

  /// 2nd vegas to load state into
  auto veg_alt = Basin(2);
  veg_alt.set_options({.verbosity = 0});
  veg_alt.read_state_stream(ss);
  REQUIRE(veg.hash().value() == veg_alt.hash().value());

  /// another warmup:  no adaption; save data
  ss.str("");  // clear content
  ss.clear();  // clear flags
  veg.integrate(test_integrand, {.neval = 1000, .niter = 10, .adapt = false});
  veg.write_data_stream(ss);
  veg.adapt();

  /// read data & adapt separately
  veg_alt.read_data_stream(ss);
  veg_alt.adapt();
  REQUIRE(veg.hash().value() == veg_alt.hash().value());
}

TEST_CASE("Basin rejects invalid adaptation parameters", "[basin]") {
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  constexpr double inf = std::numeric_limits<double>::infinity();
  auto bas = Basin(2);
  for (const double alpha : {-1e-3, nan, inf}) {
    REQUIRE_THROWS_AS(bas.set_alpha(alpha), std::invalid_argument);
  }
  for (const double weight : {0.5, 1.0, 1.99, nan, inf}) {
    REQUIRE_THROWS_AS(bas.set_weight_smooth(weight), std::invalid_argument);
  }
  for (const double gain : {-1e-3, nan}) {
    REQUIRE_THROWS_AS(bas.set_min_gain(gain), std::invalid_argument);
  }
  REQUIRE(bas.alpha() == 0.75);
  REQUIRE(bas.weight_smooth() == 6.0);
  REQUIRE(bas.min_gain() == 0.03);
  bas.set_alpha(0.0);
  bas.set_weight_smooth(2.0);
  bas.set_min_gain(inf);
  REQUIRE(bas.alpha() == 0.0);
  REQUIRE(bas.weight_smooth() == 2.0);
  REQUIRE(bas.min_gain() == inf);
}

namespace {
using S = Basin<>::size_type;
using U = Basin<>::count_type;

template <typename T = double>
struct OrderedBasin : Basin<kakuhen::util::num_traits_t<T>> {
  using Base = Basin<kakuhen::util::num_traits_t<T>>;
  using Base::grid_;
  using Base::ndim;
  using Base::order_;
  using Base::result_;
  using Base::uniform_distribution_;
  using typename Base::grid_acc_type;

  explicit OrderedBasin(S ndim, S ndiv1 = 8, S ndiv2 = 16) : Base(ndim, ndiv1, ndiv2) {
    this->set_options({.verbosity = 0});
  }

  std::vector<S> parents() const {
    std::vector<S> parent(ndim());
    for (S i = 0; i < ndim(); ++i)
      parent[order_(i, 1)] = order_(i, 0);
    return parent;
  }

  /// sample dimension i given parent[i] (roots are their own parent)
  void set_parents(const std::vector<S>& parent) {
    for (S i = 0; i < ndim(); ++i) {
      order_(i, 0) = parent[i];
      order_(i, 1) = i;
    }
    std::stringstream ss;  // the round trip sorts the order and rebuilds the sampling views
    this->write_state_stream(ss);
    this->read_state_stream(ss);
  }

  auto collected_cells() const {
    using namespace kakuhen::util::serialize;
    std::stringstream ss;
    this->write_data_stream(ss);
    S dims[3];
    kakuhen::util::HashValue_t hash;
    IntegralAccumulator<T, U> sums;
    U count;
    for (S& d : dims)
      deserialize_one(ss, d);
    deserialize_one(ss, hash);
    sums.deserialize(ss);
    deserialize_one(ss, count);
    kakuhen::ndarray::NDArray<grid_acc_type, S> cells;
    cells.deserialize(ss);
    return std::pair{count, std::move(cells)};
  }

  /// local factor moment E[h^2 w_ab/w_b], without adapt's additional division by N
  double predicted_moment(S a, S b) const {
    const auto [count, cells] = collected_cells();
    // G times the weight of its own cell: ndiv0 bins on the diagonal, ndiv2 per row else
    const S nbin = a == b ? this->ndiv0() : this->ndiv2();
    const T* edge = &grid_(a, b, 0, 0);
    const grid_acc_type* cell = &cells(a, b, 0, 0);
    double sum = 0;
    for (S k = 0; k < this->ndiv0(); ++k) {
      const T low = k % nbin > 0 ? edge[k - 1] : T(0);
      sum += double(cell[k].value() / T(count) * (T(nbin) * (edge[k] - low)));
    }
    return sum;
  }
};

/// second moment of a single-iteration result
double second_moment(const Basin<>::result_type& res, U n) {
  return res.value() * res.value() + double(n - 1) * res.error() * res.error();
}

auto ridge(double width) {
  return [width](const Point<>& p) {
    const double d = (p.x[1] - p.x[0]) / width;
    return std::exp(-0.5 * d * d) + 0.1;
  };
}

// Controlled 2x2 grids: all points visit row/bin zero, whose weight is 2*edge.
void set_order_weight(OrderedBasin<>& bas, S a, S b, double weight) {
  for (S row = 0; row < bas.ndiv1(); ++row)
    bas.grid_(a, b, row, 0) = weight / 2;
}

void collect_order_batch(OrderedBasin<>& bas) {
  bas.set_alpha(0.0);  // keep the boundaries fixed to isolate the order decision
  bas.uniform_distribution_ = std::uniform_real_distribution<double>(0.125, 0.125);
  bas.integrate([](const Point<>& p) { return (p.sample_index % 2 == 0 ? 0.0 : 2.0) / p.weight; },
                {.neval = 4, .niter = 1, .adapt = false, .verbosity = 0});
}
}  // namespace

TEST_CASE("Basin order decisions use all collected batches", "[basin]") {
  OrderedBasin whole(2, 2, 2), batched(2, 2, 2);
  for (auto* bas : {&whole, &batched}) {
    for (S row = 0; row < 2; ++row)
      bas->grid_(0, 1, row, 0) = 0.375;
    bas->set_parents({0, 1});
    bas->uniform_distribution_ = std::uniform_real_distribution<double>(0.125, 0.125);
  }
  // The three contributions are 0, 2, 2. Using only the last batch's mean (2)
  // with their pooled second moment (8/3) would give a negative variance.
  const auto f = [](const Point<>& p) { return p.sample_index == 0 ? 0.0 : 2.0; };
  whole.integrate(f, {.neval = 3, .niter = 1, .adapt = false, .verbosity = 0});
  const auto first = batched.integrate(f, {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});
  REQUIRE(first.count() == 2);
  REQUIRE(first.value() == 1.0);

  bool interrupted = false;
  SECTION("separate calls") {}
  SECTION("read data and continue collecting") {
    std::stringstream data;
    batched.write_data_stream(data);
    batched.clear_data();
    batched.read_data_stream(data);
  }
  SECTION("merge data and continue collecting") {
    std::stringstream data;
    batched.write_data_stream(data);
    batched.accumulate_data_stream(data);
    whole.integrate(f, {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});
  }
  SECTION("an exception keeps the completed samples") {
    interrupted = true;
    const auto throwing = [](const Point<>& p) {
      if (p.sample_index == 1) throw std::runtime_error("stop");
      return 2.0;
    };
    REQUIRE_THROWS_AS(
        batched.integrate(throwing, {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0}),
        std::runtime_error);
  }
  if (!interrupted) {
    const auto last = batched.integrate([](const Point<>&) { return 2.0; },
                                        {.neval = 1, .niter = 1, .adapt = false, .verbosity = 0});
    REQUIRE(last.count() == 1);
    REQUIRE(last.value() == 2.0);
  }
  REQUIRE(batched.result_.count() == whole.result_.count());
  REQUIRE(batched.result_.value() == Approx(whole.result_.value()));
  whole.adapt();
  batched.adapt();
  REQUIRE(whole.nblocks() == 1);
  REQUIRE(batched.map_hash() == whole.map_hash());
}

TEMPLATE_TEST_CASE("Basin collects large finite grid data", "[basin]", double, float) {
  using T = TestType;
  OrderedBasin<T> bas(1, 2, 2), loaded(1, 2, 2);
  // uniform weights are 1; a large contribution whose square still fits T
  const T contribution = std::ldexp(T(1), std::numeric_limits<T>::max_exponent * 25 / 64);
  const auto result = bas.integrate([=](const auto&) { return contribution; },
                                    {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});
  REQUIRE(result.count() == 2);
  REQUIRE(bas.result_.is_finite());
  REQUIRE(bas.predicted_moment(0, 0) == Approx(double(contribution) * double(contribution)));

  std::stringstream data;
  bas.write_data_stream(data);
  REQUIRE_NOTHROW(loaded.read_data_stream(data));
  REQUIRE_NOTHROW(bas.adapt());
  REQUIRE_NOTHROW(loaded.adapt());
  REQUIRE(bas.map_hash() == loaded.map_hash());
  REQUIRE(std::ranges::all_of(bas.grid_, [](T x) { return std::isfinite(x); }));
}

TEMPLATE_TEST_CASE("Basin rejects non-finite grid data before changing the sampling map", "[basin]",
                   double, float) {
  using T = TestType;
  OrderedBasin<T> bas(2, 2, 2), loaded(2, 2, 2);
  const int exponent = std::numeric_limits<T>::max_exponent;
  const T w = std::ldexp(T(1), -3 * exponent / 4);
  const T contribution = std::ldexp(T(1), exponent / 8);
  bas.grid_(1, 1, 0, 0) = w / T(4);
  bas.set_parents({0, 1});
  bas.uniform_distribution_ = std::uniform_real_distribution<T>(T(0.125), T(0.125));
  bas.integrate([=](const auto& p) { return contribution / p.weight; },
                {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});

  // The first diagonal block is finite and could refine; a later block contains overflowing G.
  REQUIRE(bas.result_.is_finite());
  REQUIRE(std::isfinite(bas.predicted_moment(0, 0)));
  REQUIRE_FALSE(std::isfinite(bas.predicted_moment(1, 1)));
  const auto map_before = bas.map_hash();
  std::stringstream data_before;
  bas.write_data_stream(data_before);

  REQUIRE_THROWS_AS(bas.adapt(), std::overflow_error);
  REQUIRE(bas.map_hash() == map_before);
  std::stringstream data_after;
  bas.write_data_stream(data_after);
  REQUIRE(data_after.str() == data_before.str());

  // The receiver has the same map, so rejection is due to the invalid statistics.
  std::stringstream state;
  bas.write_state_stream(state);
  loaded.read_state_stream(state);
  std::stringstream incoming(data_before.str());
  REQUIRE_THROWS_AS(loaded.read_data_stream(incoming), std::runtime_error);

  bas.clear_data();
  REQUIRE_NOTHROW(bas.integrate([](const auto&) { return T(1); },
                                {.neval = 8, .niter = 1, .adapt = true, .verbosity = 0}));
  REQUIRE(std::ranges::all_of(bas.grid_, [](T x) { return std::isfinite(x); }));
}

TEMPLATE_TEST_CASE("Basin rejects overflowing cell sums at adaptation", "[basin]", double, float) {
  using T = TestType;
  OrderedBasin<T> bas(1, 2, 2);
  bas.uniform_distribution_ = std::uniform_real_distribution<T>(T(0.125), T(0.125));
  const T contribution = std::sqrt(std::numeric_limits<T>::max() * T(0.75));
  REQUIRE(std::isfinite(contribution * contribution));
  bas.integrate([=](const auto&) { return contribution; },
                {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});
  REQUIRE_FALSE(std::isfinite(bas.predicted_moment(0, 0)));
  REQUIRE_THROWS_AS(bas.adapt(), std::overflow_error);
}

TEST_CASE("Basin collects finite local data with subnormal conditional weights", "[basin]") {
  OrderedBasin bas(2, 2, 2);
  const int min_subnormal_exponent =
      std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits;
  const double y_weight = std::ldexp(1.0, min_subnormal_exponent + 2);
  const double contribution = std::ldexp(1.0, min_subnormal_exponent / 2);
  bas.grid_(0, 0, 0, 0) = 0.5;  // sampled x weight is 2, so G_x rounds down to zero
  for (S row = 0; row < 2; ++row)
    bas.grid_(0, 1, row, 0) = y_weight / 2;
  bas.set_parents({0, 0});
  bas.uniform_distribution_ = std::uniform_real_distribution<double>(0.125, 0.125);
  bas.integrate([=](const Point<>& p) { return contribution / p.weight; },
                {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});

  // G_x underflows to zero, G_y stays finite
  REQUIRE(bas.result_.is_finite());
  REQUIRE(std::isfinite(bas.predicted_moment(0, 1)));
  REQUIRE(bas.predicted_moment(1, 0) == 0);
  REQUIRE_NOTHROW(bas.adapt());
  REQUIRE(std::ranges::all_of(bas.grid_, [](double x) { return std::isfinite(x); }));
}

TEST_CASE("Basin active moments use the sampled conditional bin at a boundary", "[basin]") {
  OrderedBasin bas(2, 2, 2);
  double edge = 0.25;
  U n = 2;
  SECTION("an exact bin boundary") {
    bas.uniform_distribution_ = std::uniform_real_distribution<double>(0.5, 0.5);
  }
  SECTION("default RNG rounds a narrow cell to its lower edge") {
    edge = std::nextafter(1.0, 0.0);
    n = 10000;
    bas.set_seed(7);
  }
  for (S row = 0; row < 2; ++row)
    bas.grid_(0, 1, row, 0) = edge;
  bas.set_parents({0, 0});
  U boundary_hits = 0;
  const auto result = bas.integrate(
      [&](const Point<>& p) {
        if (p.x[1] == edge) ++boundary_hits;
        return edge > 0.99 && p.x[1] >= edge ? 1 / std::sqrt(1 - edge) : 1.0;
      },
      {.neval = n, .niter = 1, .adapt = false, .verbosity = 0});
  REQUIRE(boundary_hits > 0);
  REQUIRE(bas.predicted_moment(0, 1) == Approx(second_moment(result, n)));
}

TEST_CASE("Basin passive collection selects the first containing bin on nonuniform grids",
          "[basin]") {
  OrderedBasin bas(2, 2, 4);
  const double diagonal[] = {0.125, 0.25, 0.25, 0.75, 0.8125, 0.875, 0.9375, 1.0};
  const double conditional[][4] = {{0.0, 0.25, 0.25, 1.0}, {0.1, 0.2, 0.3, 1.0}};
  for (S a = 0; a < 2; ++a)
    for (S row = 0; row < 2; ++row)
      for (S k = 0; k < 4; ++k) {
        bas.grid_(a, a, row, k) = diagonal[row * 4 + k];
        bas.grid_(a, 1 - a, row, k) = conditional[row][k];
      }
  bas.set_parents({0, 1});

  // Cover repeated edges, exact boundaries, zero-width sampled cells and both endpoints.
  kakuhen::ndarray::NDArray<U, S> expected({2, 2, 2, 4});
  std::ranges::fill(expected, U(0));
  const double draws[] = {
      0.0, 0.125, 0.1875, 0.25, 0.375, 0.4375, 0.5, 0.75, std::nextafter(1.0, 0.0)};
  for (const double u : draws) {
    bas.uniform_distribution_ = std::uniform_real_distribution<double>(u, u);
    const S sampled_bin = S(u * bas.ndiv0());
    bas.integrate(
        [&](const Point<>& point) {
          for (S a = 0; a < 2; ++a) {
            ++expected(a, a, sampled_bin / 4, sampled_bin % 4);
            const S b = 1 - a, row = sampled_bin / 4;
            const double* edges = &bas.grid_(a, b, row, 0);
            const S bin = S(std::lower_bound(edges, edges + 4, point.x[b]) - edges);
            ++expected(a, b, row, bin);
          }
          return 1.0;
        },
        {.neval = 2, .niter = 1, .adapt = false});
  }
  const auto [count, cells] = bas.collected_cells();
  REQUIRE(count == 2 * std::size(draws));
  for (S a = 0; a < 2; ++a)
    for (S b = 0; b < 2; ++b)
      for (S row = 0; row < 2; ++row)
        for (S k = 0; k < 4; ++k) {
          CAPTURE(a, b, row, k);
          REQUIRE(cells(a, b, row, k).count() == expected(a, b, row, k));
        }
}

TEST_CASE("Basin collects with a narrow size type", "[basin]") {
  Basin<kakuhen::util::num_traits_t<double, std::uint16_t>> bas(8, 8, 16);
  const auto result = bas.integrate([](const auto&) { return 1.0; },
                                    {.neval = 2, .niter = 1, .adapt = false, .verbosity = 0});
  REQUIRE(result.value() == Approx(1.0));
}

TEST_CASE("Basin rejects a grid whose size exceeds its size type", "[basin]") {
  using SmallBasin = Basin<kakuhen::util::num_traits_t<double, std::uint16_t>>;
  REQUIRE_THROWS_AS(SmallBasin(1, 65535, 65535), std::runtime_error);
}

TEST_CASE("Basin maps roots into their sampled cell at a rounding boundary", "[basin]") {
  OrderedBasin bas(1, 3, 4);
  // 12 u rounds up to 5: an fma of 12 u - 5 would be negative and put x below the cell
  const double u = std::nextafter(5.0 / 12.0, 0.0);
  REQUIRE(S(u * 12.0) == 5);
  bas.uniform_distribution_ = std::uniform_real_distribution<double>(u, u);
  double x = -1.0;
  bas.integrate(
      [&](const Point<>& point) {
        x = point.x[0];
        return 1.0;
      },
      {.neval = 1, .niter = 1, .adapt = false});
  REQUIRE(x >= bas.grid_(0, 0, 1, 0));  // lower edge of cell 5: row 1, edge 0
  REQUIRE(x <= bas.grid_(0, 0, 1, 1));
}

TEST_CASE("Basin keeps grid rows when refinement weights underflow", "[basin]") {
  OrderedBasin bas(2, 2, 2);
  SECTION("positive grid widths") {}
  SECTION("rounded zero-width coarse cell") {
    bas.grid_(0, 0, 0, 0) = 0.0;
    bas.grid_(0, 0, 0, 1) = 0.0;
    bas.set_parents({0, 1});
  }
  bas.set_alpha(1e6);
  bas.set_min_gain(std::numeric_limits<double>::infinity());
  const auto grid_hash = bas.hash().value();
  bas.integrate(ridge(0.15), {.neval = 100, .niter = 1, .adapt = true, .verbosity = 0});
  REQUIRE(bas.hash().value() == grid_hash);
}

TEST_CASE("Basin data predict the variance of other sampling orders", "[basin]") {
  constexpr U n = 400000;
  const auto f = ridge(0.15);
  OrderedBasin bas(2, 4, 8);
  bas.set_seed(7);
  bas.set_min_gain(std::numeric_limits<double>::infinity());  // keep the order fixed
  bas.integrate(f, {.neval = 20000, .niter = 5, .adapt = true, .verbosity = 0});

  // collect on `from`, predict block (a, b), then sample with `to` on the same grid
  const auto check = [&](const std::vector<S>& from, S a, S b, const std::vector<S>& to) {
    CAPTURE(from, a, b);
    bas.set_parents(from);
    bas.integrate(f, {.neval = n, .niter = 1, .adapt = false, .verbosity = 0});
    const double moment = bas.predicted_moment(a, b);
    bas.set_parents(to);
    const auto res = bas.integrate(f, {.neval = n, .niter = 1, .frozen = true, .verbosity = 0});
    // compare variances, i.e. second moments minus I^2
    const double i2 = res.value() * res.value();
    REQUIRE(moment - i2 == Approx(second_moment(res, n) - i2).epsilon(0.07));
  };
  check({0, 1}, 0, 1, {0, 0});  // passive row
  check({0, 0}, 1, 1, {0, 1});  // passive diagonal
}

TEST_CASE("Basin predicts reparenting within and between sampling trees", "[basin]") {
  constexpr U n = 100000;
  OrderedBasin bas(3, 2, 2);
  bas.set_seed(42);
  for (S a = 0; a < 3; ++a) {
    bas.grid_(a, a, 0, 0) = 0.125;
    for (S b = 0; b < 3; ++b) {
      if (a == b) continue;
      bas.grid_(a, b, 0, 0) = 0.25;
      bas.grid_(a, b, 1, 0) = 0.75;
    }
  }
  const auto f = [](const Point<>& p) { return (1 + p.x[0]) * (0.2 + p.x[1] * p.x[2]); };
  const auto check = [&](const std::vector<S>& from, S a, S b, const std::vector<S>& to) {
    CAPTURE(from, a, b, to);
    bas.set_parents(from);
    bas.integrate(f, {.neval = n, .niter = 1, .adapt = false, .verbosity = 0});
    const double predicted = bas.predicted_moment(a, b);
    bas.set_parents(to);
    const auto result = bas.integrate(f, {.neval = n, .niter = 1, .frozen = true, .verbosity = 0});
    REQUIRE(predicted == Approx(second_moment(result, n)).epsilon(0.03));
  };
  check({0, 0, 1}, 1, 1, {0, 1, 1});  // detach a subtree
  check({0, 0, 0}, 1, 2, {0, 0, 1});  // attach to a sibling
  check({0, 0, 2}, 2, 1, {0, 2, 2});  // attach to another root
}

TEST_CASE("Basin applies a forest only above the total local gain threshold", "[basin]") {
  for (double threshold : {std::nextafter(0.5, 0.0), 0.5, std::nextafter(0.5, 1.0)}) {
    CAPTURE(threshold);
    OrderedBasin bas(3, 2, 2);
    for (S row = 0; row < 2; ++row) {
      bas.grid_(1, 2, row, 0) = 1.0 / 3.0;
      bas.grid_(2, 0, row, 0) = 0.125;
    }
    bas.set_parents({0, 0, 1});
    bas.set_alpha(0.0);  // hold the grid fixed while testing the order decision
    bas.set_min_gain(threshold);
    bas.uniform_distribution_ = std::uniform_real_distribution<double>(0.125, 0.125);
    const auto result = bas.integrate(
        [](const Point<>& p) { return (p.sample_index % 2 == 0 ? 0.0 : 2.0) / p.weight; },
        {.neval = 4, .niter = 1, .adapt = false, .verbosity = 0});
    REQUIRE(result.value() == 1.0);
    REQUIRE(second_moment(result, 4) == Approx(2.0));
    REQUIRE(bas.predicted_moment(2, 0) == 0.5);  // local gain 1.5 for changing 0
    REQUIRE(bas.predicted_moment(2, 2) == 3.0);  // local cost 1 for making 2 a root
    const auto grid_hash = bas.hash().value();
    bas.adapt();
    REQUIRE(bas.hash().value() == grid_hash);
    REQUIRE(bas.parents() == (threshold < 0.5 ? std::vector<S>{2, 0, 2} : std::vector<S>{0, 0, 1}));
  }
}

TEST_CASE("Basin updates multiple dimensions from one batch above the local gain threshold",
          "[basin][order_moves]") {
  for (double threshold : {0.0, 0.03, std::nextafter(0.5, 0.0), 0.5, std::nextafter(0.5, 1.0),
                           std::numeric_limits<double>::infinity()}) {
    CAPTURE(threshold);
    OrderedBasin bas(4, 2, 2);
    set_order_weight(bas, 0, 1, 0.5);
    set_order_weight(bas, 2, 3, 0.75);
    bas.set_parents({0, 1, 2, 3});
    bas.set_min_gain(threshold);
    collect_order_batch(bas);
    REQUIRE(bas.predicted_moment(0, 1) == 1.0);  // current M = 2, variance = 1
    REQUIRE(bas.predicted_moment(2, 3) == 1.5);  // second local gain = 0.5
    const auto grid_hash = bas.hash().value();
    bas.adapt();
    REQUIRE(bas.hash().value() == grid_hash);
    REQUIRE(bas.parents() == (threshold < 0.5   ? std::vector<S>{0, 0, 2, 2}
                              : threshold < 1.0 ? std::vector<S>{0, 0, 2, 3}
                                                : std::vector<S>{0, 1, 2, 3}));
    REQUIRE(bas.result_.count() == 0);
  }
}

TEST_CASE("Basin rejects a cyclic move and considers the next parent choice",
          "[basin][order_moves]") {
  OrderedBasin bas(4, 2, 2);
  set_order_weight(bas, 0, 1, 0.25);   // highest gain: 0 -> 1
  set_order_weight(bas, 1, 2, 0.5);    // can extend the chain despite 1's new parent
  set_order_weight(bas, 2, 0, 0.75);   // would close the cycle 0 -> 1 -> 2 -> 0
  set_order_weight(bas, 3, 0, 0.875);  // a lower gain, valid alternative for 0
  bas.set_parents({0, 1, 2, 3});
  collect_order_batch(bas);
  REQUIRE_NOTHROW(bas.adapt());
  REQUIRE(bas.parents() == std::vector<S>{3, 0, 1, 3});
  REQUIRE(bas.nblocks() == 1);
}

TEST_CASE("Basin resolves equal local gains in dimension scan order", "[basin][order_moves]") {
  OrderedBasin bas(3, 2, 2);
  set_order_weight(bas, 1, 0, 0.5);
  set_order_weight(bas, 2, 0, 0.5);
  bas.set_parents({0, 1, 2});
  collect_order_batch(bas);
  REQUIRE(bas.predicted_moment(1, 0) == bas.predicted_moment(2, 0));
  bas.adapt();
  REQUIRE(bas.parents() == std::vector<S>{1, 1, 2});
}

TEST_CASE("Basin can attach parents after selecting reversed dependencies",
          "[basin][order_moves]") {
  OrderedBasin bas(5, 2, 2);
  set_order_weight(bas, 2, 0, 0.25);  // select 2 -> 0 without fixing 2 as a root
  set_order_weight(bas, 1, 0, 0.5);   // cannot overwrite 0's selected parent
  set_order_weight(bas, 2, 1, 0.5);   // another child may use the same parent
  set_order_weight(bas, 3, 2, 0.5);   // 2 itself may then receive a parent
  set_order_weight(bas, 3, 4, 0.75);  // compatible independent move
  bas.set_parents({0, 0, 1, 3, 4});
  collect_order_batch(bas);
  REQUIRE_NOTHROW(bas.adapt());
  REQUIRE(bas.parents() == std::vector<S>{2, 2, 3, 3, 3});
}

TEST_CASE("Basin retains compatible original parents after reversed choices",
          "[basin][order_moves]") {
  OrderedBasin bas(4, 2, 2);
  set_order_weight(bas, 3, 1, 0.25);  // first move removes 2 from 0's descendants
  set_order_weight(bas, 2, 0, 0.5);   // can retain 1 -> 2 in the completed forest
  bas.set_parents({0, 0, 1, 3});
  collect_order_batch(bas);
  REQUIRE_NOTHROW(bas.adapt());
  REQUIRE(bas.parents() == std::vector<S>{2, 3, 1, 3});
}

TEST_CASE("Basin forms a reversed chain in either local gain ranking", "[basin][order_moves]") {
  for (bool reverse_first : {true, false}) {
    CAPTURE(reverse_first);
    OrderedBasin bas(3, 2, 2);
    set_order_weight(bas, 0, 1, 0.5);  // old b -> a; making a a root has a local cost
    set_order_weight(bas, 1, 0, reverse_first ? 0.5 : 0.75);
    set_order_weight(bas, 2, 1, reverse_first ? 0.375 : 0.125);
    bas.set_parents({0, 0, 2});
    collect_order_batch(bas);
    bas.adapt();
    REQUIRE(bas.parents() == std::vector<S>{1, 2, 2});  // c -> a -> b
  }
}

TEST_CASE("Basin accounts for a forced root's negative local gain", "[basin][order_moves]") {
  OrderedBasin bas(2, 2, 2);
  set_order_weight(bas, 0, 1, 0.5);
  set_order_weight(bas, 1, 0, 0.5);
  bas.set_parents({0, 0});
  collect_order_batch(bas);
  REQUIRE(bas.predicted_moment(1, 0) == 1.0);  // raw G: local gain +1
  REQUIRE(bas.predicted_moment(1, 1) == 4.0);  // required root: local gain -2
  bas.adapt();
  REQUIRE(bas.parents() == std::vector<S>{0, 0});
}

TEST_CASE("Basin keeps a better forest before a harmful later choice", "[basin][order_moves]") {
  OrderedBasin bas(4, 2, 2);
  set_order_weight(bas, 0, 1, 0.5);
  set_order_weight(bas, 1, 0, 0.5);   // gain +1, but its required root costs 2
  set_order_weight(bas, 2, 3, 0.25);  // independent gain +1.5
  bas.set_parents({0, 0, 2, 3});
  collect_order_batch(bas);
  bas.adapt();
  REQUIRE(bas.parents() == std::vector<S>{0, 0, 2, 2});
}

TEST_CASE("Basin can return multiple dimensions to diagonal sampling in one adaptation",
          "[basin][order_moves]") {
  OrderedBasin bas(3, 2, 2);
  set_order_weight(bas, 0, 1, 1.5);
  set_order_weight(bas, 0, 2, 1.5);
  for (const auto [a, b] : {std::pair{1, 0}, {2, 0}, {1, 2}, {2, 1}})
    set_order_weight(bas, S(a), S(b), 1.75);  // only the two diagonal alternatives improve
  bas.set_parents({0, 0, 0});
  collect_order_batch(bas);
  REQUIRE_NOTHROW(bas.adapt());
  REQUIRE(bas.parents() == std::vector<S>{0, 1, 2});
  REQUIRE(bas.nblocks() == 3);
}

TEST_CASE("Basin adopts a conditional order that pays off and keeps it", "[basin]") {
  const auto f = ridge(0.05);
  const auto run = [&](double min_gain) {
    OrderedBasin bas(2);
    bas.set_seed(11);
    bas.set_min_gain(min_gain);
    std::vector<std::vector<S>> orders;
    double error = 0;
    for (int it = 0; it < 12; ++it) {
      error = bas.integrate(f, {.neval = 20000, .niter = 1, .adapt = true, .verbosity = 0}).error();
      orders.push_back(bas.parents());
    }
    return std::pair{orders, error};
  };
  const auto [orders, error] = run(0.05);
  REQUIRE(orders[4][0] == orders[4][1]);  // one root
  REQUIRE(
      std::all_of(orders.begin() + 4, orders.end(), [&](const auto& p) { return p == orders[4]; }));
  const auto [diag_orders, diag_error] = run(std::numeric_limits<double>::infinity());
  REQUIRE(diag_orders.back() == std::vector<S>{0, 1});
  REQUIRE(error < 0.5 * diag_error);
}

TEST_CASE("Basin settles on the diagonal order for a factorized integrand", "[basin]") {
  Basin<> bas(3);
  bas.set_seed(5);
  const auto f = [](const Point<>& p) {
    return (1 + 9 * p.x[0] * p.x[0]) * (0.2 + p.x[1]) * std::exp(-2 * p.x[2]);
  };
  // conditional rows adapt faster at first, so the warmup may borrow them
  for (int it = 0; it < 20; ++it) {
    bas.integrate(f, {.neval = 20000, .niter = 1, .adapt = true, .verbosity = 0});
    if (it >= 12) REQUIRE(bas.nblocks() == 3);
  }
}

TEST_CASE("Basin data require the same sampling order", "[basin]") {
  OrderedBasin worker(2, 2, 4), other(2, 2, 4);
  worker.integrate(test_integrand, {.neval = 100, .niter = 1, .adapt = false, .verbosity = 0});
  std::stringstream ss;
  worker.write_data_stream(ss);
  other.set_parents({0, 0});
  REQUIRE(worker.hash().value() == other.hash().value());  // same grid
  REQUIRE_THROWS_AS(other.read_data_stream(ss), std::runtime_error);
}
