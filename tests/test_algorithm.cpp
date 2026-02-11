#include "kakuhen/util/algorithm.h"
#include <algorithm>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

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
