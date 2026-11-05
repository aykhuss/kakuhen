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
using Readonly = NDView<const int, S>;

template <typename H>
concept Writable = requires(H& h) {
  h(0, 0) = 9;
  h.data()[0] = 9;
  h.fill(9);
};

template <typename H>
concept Fillable = requires(H& h) { h.fill(9); };

template <typename H, typename Expected>
constexpr bool factory_types() {
  static_assert(std::same_as<decltype(std::declval<H&>().view()), Expected>);
  static_assert(std::same_as<decltype(std::declval<H&>().slice({{}, {}})), Expected>);
  static_assert(
      std::same_as<decltype(std::declval<H&>().slice(std::declval<std::span<const Slice<S>>>())),
                   Expected>);
  static_assert(
      std::same_as<decltype(std::declval<H&>().slice(std::declval<const std::vector<Slice<S>>&>())),
                   Expected>);
  static_assert(std::same_as<decltype(std::declval<H&>().reshape({2, 2})), Expected>);
  static_assert(
      std::same_as<decltype(std::declval<H&>().reshape(std::declval<std::span<const S>>())),
                   Expected>);
  static_assert(
      std::same_as<decltype(std::declval<H&>().reshape(std::declval<const std::vector<S>&>())),
                   Expected>);
  static_assert(std::same_as<decltype(std::declval<H&>().diagonal(0, 1)), Expected>);
  return true;
}

static_assert(factory_types<Owner, View>());
static_assert(factory_types<const Owner, Readonly>());
static_assert(Writable<View> && Writable<const View>);
static_assert(!Writable<Readonly> && !Fillable<Readonly> && !Fillable<const Readonly>);
static_assert(std::same_as<decltype(std::declval<const View&>()(0, 0)), int&>);
static_assert(std::same_as<decltype(std::declval<const Readonly&>()(0, 0)), const int&>);
static_assert(std::same_as<decltype(std::declval<const View&>().data()), int*>);
static_assert(std::same_as<decltype(std::declval<Readonly&>().data()), const int*>);
static_assert(std::same_as<decltype(std::declval<const View&>().slice({{}, {}})), View>);
static_assert(std::same_as<decltype(std::declval<const Readonly&>().reshape({4})), Readonly>);
static_assert(std::same_as<decltype(std::declval<const Readonly&>().diagonal(0, 1)), Readonly>);
static_assert(std::is_constructible_v<View, Owner&>);
static_assert(!std::is_constructible_v<View, const Owner&>);
static_assert(std::is_convertible_v<Owner&, Readonly>);
static_assert(std::is_convertible_v<const Owner&, Readonly>);
static_assert(std::is_convertible_v<const View&, Readonly>);
static_assert(std::is_assignable_v<Readonly&, const View&>);
static_assert(!std::is_constructible_v<View, const Readonly&>);
static_assert(!std::is_constructible_v<View, Readonly&&>);
static_assert(!std::is_assignable_v<View&, Readonly>);
static_assert(!std::is_assignable_v<const View&, View>);
static_assert(std::is_nothrow_move_constructible_v<View>);
static_assert(std::is_nothrow_move_assignable_v<View>);
static_assert(std::is_nothrow_constructible_v<Readonly, View&&>);
static_assert(std::is_nothrow_assignable_v<Readonly&, View&&>);

template <typename Index>
void check_lifetimes() {
  using Mutable = NDView<int, Index>;
  using Const = NDView<const int, Index>;
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
    Const converted, converted_move;
    {
      auto source = owner.slice(slices);
      copied = Mutable(source);
      assigned = source;
      converted = source;
      REQUIRE(copied.shape().data() != source.shape().data());
      REQUIRE(converted.strides().data() != source.strides().data());
      moved = Mutable(std::move(source));
      REQUIRE(source.empty());
      REQUIRE(source.shape().empty());
      source = owner.slice(slices);
      converted_move = std::move(source);
      REQUIRE(source.strides().empty());
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
    check(converted);
    check(converted_move);
    copied.fill(12);
    REQUIRE(owner[1] == 12);
    REQUIRE(owner[7] == 12);
    REQUIRE(owner[2] == 2);
    REQUIRE(converted.data()[0] == 12);
    assigned = scratch.view();  // heap-to-inline assignment
    REQUIRE(assigned(1, 1) == 5);
  }
}

// enumerate the selected offsets independently of fill's run traversal and check the
// whole backing buffer, so writes outside the selection cannot pass unnoticed
void check_fill(std::vector<int>& backing, const NDView<int, std::int32_t>& view) {
  auto expected = backing;
  for (std::int32_t flat = 0; flat < view.size(); ++flat) {
    auto remaining = flat;
    std::ptrdiff_t offset = view.data() - backing.data();
    for (std::size_t axis = view.shape().size(); axis-- > 0;) {
      offset += (remaining % view.shape()[axis]) * view.strides()[axis];
      remaining /= view.shape()[axis];
    }
    REQUIRE(offset >= 0);
    REQUIRE(static_cast<std::size_t>(offset) < expected.size());
    expected[static_cast<std::size_t>(offset)] = 42;
  }
  view.fill(42);
  REQUIRE(backing == expected);
}
}  // namespace

TEST_CASE("NDView metadata survives copying moving and source reuse", "[ndview]") {
  check_lifetimes<std::uint16_t>();
  check_lifetimes<std::uint32_t>();
  check_lifetimes<std::uint64_t>();
}

TEST_CASE("NDView deduces element and index types from mutable and const owners", "[ndview]") {
  auto check = []<typename Index>() {
    NDArray<int, Index> owner({2, 3});
    owner.fill(5);
    NDView mutable_view(owner);
    NDView readonly_view(std::as_const(owner));
    NDView mutable_copy = owner;
    NDView readonly_copy = std::as_const(owner);
    STATIC_REQUIRE(std::same_as<decltype(mutable_view), NDView<int, Index>>);
    STATIC_REQUIRE(std::same_as<decltype(readonly_view), NDView<const int, Index>>);
    STATIC_REQUIRE(std::same_as<decltype(mutable_copy), NDView<int, Index>>);
    STATIC_REQUIRE(std::same_as<decltype(readonly_copy), NDView<const int, Index>>);
    mutable_view(1, 2) = 17;
    REQUIRE(owner(1, 2) == 17);
    REQUIRE(readonly_view(1, 2) == 17);
    REQUIRE(mutable_copy(1, 2) == 17);
    REQUIRE(readonly_copy(1, 2) == 17);
  };
  check.operator()<std::uint16_t>();
  check.operator()<std::uint32_t>();
  check.operator()<std::uint64_t>();
}

TEST_CASE("NDArray constness controls elements while NDView constness controls the handle",
          "[ndview]") {
  Owner owner({3, 3});
  owner.fill(0);
  const View handle = owner.view();
  handle(1, 2) = 17;
  handle.data()[0] = 4;
  handle.diagonal(0, 1).fill(8);
  REQUIRE(owner(1, 2) == 17);
  REQUIRE(owner(2, 2) == 8);
  const Owner& const_owner = owner;
  Readonly readonly = const_owner;
  REQUIRE(readonly(1, 2) == 17);
  REQUIRE(const_owner.slice({{1, 2}, {2, 3}})(0, 0) == 17);
  REQUIRE(const_owner.reshape({9})(5) == 17);
  REQUIRE(const_owner.diagonal(0, 1)(2) == 8);
  owner(1, 2) = 19;
  REQUIRE(readonly(1, 2) == 19);
  const Owner empty;
  REQUIRE(Readonly(empty).empty());
}

TEST_CASE("NDView reshape counts a rank-0 view as empty like size()", "[ndview]") {
  const View rank0;
  REQUIRE(rank0.empty());
  const auto rank1 = rank0.reshape({0});  // used to trip the size assertion
  REQUIRE(rank1.ndim() == 1);
  REQUIRE(rank1.empty());
  REQUIRE(rank1.data() == nullptr);
  const auto back = rank1.reshape(std::span<const S>{});
  REQUIRE(back.ndim() == 0);
  REQUIRE(back.empty());
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

TEST_CASE("NDView fill changes only selected elements", "[ndview][fill]") {
  using I = std::int32_t;
  using kakuhen::ndarray::_;
  NDArray<int, I> owner({4, 4, 3});
  std::vector<int> backing(static_cast<std::size_t>(owner.size()));
  std::iota(backing.begin(), backing.end(), -100);
  NDView<int, I> view(backing.data(), owner.shape().data(), owner.strides().data(), 3);
  check_fill(backing, view.slice({{1, 4, 2}, {_, _, 2}, {1, 3}}));
  check_fill(backing, view.diagonal(0, 1));
  check_fill(backing, view.diagonal(1, 0).slice({{1, 4, 2}, {}}));
  check_fill(backing, view.slice({{2, 2}, {}, {}}));
  check_fill(backing, view);
  NDView<int, I> empty;
  empty.fill(42);
  REQUIRE(empty.empty());
}

TEST_CASE("NDView fill handles raw strided layouts and ranks beyond inline storage",
          "[ndview][fill]") {
  using I = std::int32_t;
  std::vector<int> backing(512);
  auto raw_case = [&](std::vector<I> shape, std::vector<I> strides, std::size_t origin = 0) {
    std::iota(backing.begin(), backing.end(), -512);
    check_fill(backing, NDView<int, I>(backing.data() + origin, shape.data(), strides.data(),
                                       static_cast<I>(shape.size())));
  };
  raw_case({3, 4}, {1, 3});  // column-major
  raw_case({3, 4}, {-7, -1}, 18);
  raw_case({3, 4}, {0, 1});  // repeated addresses
  raw_case({3, 4}, {1, 1});  // overlapping axes
  raw_case({1, 3, 1, 4}, {99, 8, 77, 1});
  raw_case({0, 3}, {3, 1});
  for (I rank = 1; rank <= 9; ++rank) {
    CAPTURE(rank);
    std::vector<I> shape(static_cast<std::size_t>(rank), 2);
    std::vector<I> strides(shape.size());
    I stride = 1;
    for (std::size_t axis = shape.size(); axis-- > 0;) {
      strides[axis] = stride;
      stride *= 2;
    }
    raw_case(shape, strides);
    shape[0] = 1;
    raw_case(shape, strides);
  }
}

TEST_CASE("NDView strided fill assigns nontrivial elements", "[ndview][fill]") {
  NDArray<std::string> owner({3, 3});
  owner.fill("unchanged");
  owner.diagonal(0, 1).fill(std::string(100, 'x'));
  for (S i = 0; i < 3; ++i)
    for (S j = 0; j < 3; ++j)
      REQUIRE(owner(i, j) == (i == j ? std::string(100, 'x') : "unchanged"));
}
