#include "kakuhen/util/algorithm.h"
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <random>
#include <vector>

namespace {

struct Case {
  std::vector<int> data;
  int value;
};

std::vector<Case> make_cases() {
  std::vector<Case> cases;

  cases.push_back({{}, 0});
  cases.push_back({{1}, 0});
  cases.push_back({{1}, 1});
  cases.push_back({{1}, 2});
  cases.push_back({{1, 1, 1}, 1});
  cases.push_back({{1, 1, 2, 2, 3}, 2});
  cases.push_back({{1, 2, 3, 4, 5}, 0});
  cases.push_back({{1, 2, 3, 4, 5}, 3});
  cases.push_back({{1, 2, 3, 4, 5}, 6});

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dist(-50, 50);

  for (int n = 0; n < 50; ++n) {
    std::vector<int> v;
    v.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      v.push_back(dist(rng));
    }
    std::sort(v.begin(), v.end());
    cases.push_back({v, dist(rng)});
    cases.push_back({v, dist(rng)});
  }

  return cases;
}

}  // namespace

TEST_CASE("algorithm::partition_point_index finds every partition across the scan cutoff",
          "[algorithm]") {
  for (std::size_t n : {0u, 1u, 31u, 32u, 33u, 64u, 128u}) {
    std::vector<int> data(n);
    for (std::size_t split = 0; split <= n; ++split) {
      // Partitioned by sign; neither group needs to be sorted.
      for (std::size_t i = 0; i < n; ++i)
        data[i] = (i < split ? 1 : -1) * (static_cast<int>(i % 3) + 1);
      const auto positive = [](int value) { return value > 0; };
      INFO("length=" << n << ", split=" << split);
      REQUIRE(kakuhen::util::algorithm::partition_point_index(data.begin(), data.end(), positive) ==
              std::partition_point(data.begin(), data.end(), positive) - data.begin());
    }
  }
}

TEST_CASE("algorithm::partition_point_index preserves lower and upper bound ties", "[algorithm]") {
  constexpr auto values = [] {
    std::array<int, 65> data{};
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = static_cast<int>(i / 3);
    return data;
  }();
  static_assert(kakuhen::util::algorithm::partition_point_index(
                    values.begin(), values.end(), [](int edge) { return edge < 11; }) == 33);
  static_assert(kakuhen::util::algorithm::partition_point_index(
                    values.begin(), values.end(), [](int edge) { return edge <= 11; }) == 36);
  static_assert(kakuhen::util::algorithm::partition_point_index(
                    values.begin(), values.begin() + 32, [](int edge) { return edge < 11; }) == 32);

  for (const auto& c : make_cases()) {
    const auto check = [&](auto first, auto last, auto comp) {
      REQUIRE(kakuhen::util::algorithm::partition_point_index(first, last, [&](int edge) {
                return comp(edge, c.value);
              }) == std::lower_bound(first, last, c.value, comp) - first);
      REQUIRE(kakuhen::util::algorithm::partition_point_index(first, last, [&](int edge) {
                return !comp(c.value, edge);
              }) == std::upper_bound(first, last, c.value, comp) - first);
    };
    check(c.data.begin(), c.data.end(), std::less<>{});
    check(c.data.rbegin(), c.data.rend(), std::greater<>{});
  }
}

TEST_CASE("algorithm::lower_bound agrees with std::lower_bound", "[algorithm]") {
  auto cases = make_cases();
  for (const auto& c : cases) {
    const auto comp = [](int a, int b) { return a < b; };

    auto it_std = std::lower_bound(c.data.begin(), c.data.end(), c.value, comp);
    auto it_kh = kakuhen::util::algorithm::lower_bound(c.data.begin(), c.data.end(), c.value, comp);

    REQUIRE(std::distance(c.data.begin(), it_std) == std::distance(c.data.begin(), it_kh));
  }
}

TEST_CASE("algorithm::upper_bound agrees with std::upper_bound", "[algorithm]") {
  auto cases = make_cases();
  for (const auto& c : cases) {
    const auto comp = [](int a, int b) { return a < b; };

    auto it_std = std::upper_bound(c.data.begin(), c.data.end(), c.value, comp);
    auto it_kh = kakuhen::util::algorithm::upper_bound(c.data.begin(), c.data.end(), c.value, comp);

    REQUIRE(std::distance(c.data.begin(), it_std) == std::distance(c.data.begin(), it_kh));
  }
}

TEST_CASE("algorithm::lower_bound_with_hint agrees with std::lower_bound", "[algorithm]") {
  auto cases = make_cases();
  for (const auto& c : cases) {
    const auto comp = [](int a, int b) { return a < b; };
    auto it_std = std::lower_bound(c.data.begin(), c.data.end(), c.value, comp);

    // exercise hints: before, after, and mid
    auto it_kh_first = kakuhen::util::algorithm::lower_bound_with_hint(
        c.data.begin(), c.data.end(), c.data.begin(), c.value, comp);
    auto it_kh_last = kakuhen::util::algorithm::lower_bound_with_hint(c.data.begin(), c.data.end(),
                                                                      c.data.end(), c.value, comp);
    auto it_kh_mid = kakuhen::util::algorithm::lower_bound_with_hint(
        c.data.begin(), c.data.end(), c.data.begin() + c.data.size() / 2, c.value, comp);

    REQUIRE(std::distance(c.data.begin(), it_std) == std::distance(c.data.begin(), it_kh_first));
    REQUIRE(std::distance(c.data.begin(), it_std) == std::distance(c.data.begin(), it_kh_last));
    REQUIRE(std::distance(c.data.begin(), it_std) == std::distance(c.data.begin(), it_kh_mid));
  }
}
