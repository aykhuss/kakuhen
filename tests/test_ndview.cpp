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

template <typename Index>
void check_lifetimes() {
  using Mutable = NDView<int, Index>;
  NDArray<int, Index> scratch({2, 2});
  scratch.fill(5);
  for (Index rank : {Index(1), Index(2), Index(4), Index(5), Index(9)}) {
    CAPTURE(rank, sizeof(Index));
    std::vector<Index> shape(rank, Index(1));
    shape[0] = 8;
    NDArray<int, Index> owner(shape);
    std::iota(owner.begin(), owner.end(), 0);
    std::vector<Slice<Index>> slices(rank);
    slices[0] = {Index(1), Index(8), Index(2)};
    Mutable copied, assigned, moved, move_assigned;
    {
      auto source = owner.slice(slices);
      copied = Mutable(source);
      assigned = source;
      REQUIRE(copied.shape().data() != source.shape().data());
      moved = Mutable(std::move(source));
      REQUIRE(source.empty());
      REQUIRE(source.shape().empty());
      source = owner.slice(slices);
      move_assigned = std::move(source);
      source = scratch.view();
      REQUIRE(source(1, 1) == 5);
    }
    auto check = [&](const auto& view) {
      REQUIRE(view.ndim() == rank);
      REQUIRE(view.size() == 4);
      REQUIRE(view.shape()[0] == 4);
      REQUIRE(view.strides()[0] == 2);
      REQUIRE(view.data() == owner.data() + 1);
      // access after the source has been reused or destroyed
      switch (rank) {
        case 1:
          REQUIRE(view(3) == 7);
          break;
        case 2:
          REQUIRE(view(3, 0) == 7);
          break;
        case 4:
          REQUIRE(view(3, 0, 0, 0) == 7);
          break;
        case 5:
          REQUIRE(view(3, 0, 0, 0, 0) == 7);
          break;
        case 9:
          REQUIRE(view(3, 0, 0, 0, 0, 0, 0, 0, 0) == 7);
          break;
      }
    };
    check(copied);
    check(assigned);
    check(moved);
    check(move_assigned);
    assigned = scratch.view();  // heap-to-inline assignment
    REQUIRE(assigned(1, 1) == 5);
  }
}

}  // namespace

TEST_CASE("NDView metadata survives copying moving and source reuse", "[ndview]") {
  check_lifetimes<std::uint16_t>();
  check_lifetimes<std::uint32_t>();
  check_lifetimes<std::uint64_t>();
}

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
