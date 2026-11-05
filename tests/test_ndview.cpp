#include "kakuhen/ndarray/ndarray.h"
#include <catch2/catch_test_macros.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using kakuhen::ndarray::NDArray;
using kakuhen::ndarray::NDView;
using kakuhen::ndarray::Slice;
using S = std::uint32_t;
using Owner = NDArray<int, S>;
using View = NDView<int, S>;

}  // namespace

TEST_CASE("NDView argument ranges preserve vector and braced calls", "[ndview]") {
  Owner owner({2, 3});
  std::iota(owner.begin(), owner.end(), 0);
  const std::vector<S> shape{3, 2};
  const std::vector<Slice<S>> slices{{}, {1, 3}};
  REQUIRE(owner.reshape(shape)(2, 1) == 5);
  REQUIRE(owner.reshape(std::span<const S>(shape))(2, 1) == 5);
  REQUIRE(owner.view().reshape(shape)(2, 1) == 5);
  REQUIRE(owner.slice(slices)(1, 1) == 5);
  REQUIRE(owner.slice(std::span<const Slice<S>>(slices))(1, 1) == 5);
  REQUIRE(owner.view().slice(slices)(1, 1) == 5);
}
