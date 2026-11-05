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
  REQUIRE(owner(1, 2) == 17);
  const Owner& const_owner = owner;
  Readonly readonly = const_owner;
  REQUIRE(readonly(1, 2) == 17);
  REQUIRE(const_owner.slice({{1, 2}, {2, 3}})(0, 0) == 17);
  REQUIRE(const_owner.reshape({9})(5) == 17);
  owner(1, 2) = 19;
  REQUIRE(readonly(1, 2) == 19);
  const Owner empty;
  REQUIRE(Readonly(empty).empty());
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
