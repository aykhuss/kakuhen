#include "kakuhen/ndarray/ndarray.h"
#include "kakuhen/util/small_vector.h"
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <span>
#include <utility>
#include <vector>

// own executable: replacing the global allocation functions affects every test in it;
// only ordinary (not over-aligned) allocations are counted
namespace {
thread_local bool tracking = false;
thread_local bool fail_next = false;
thread_local std::size_t allocations = 0;

struct AllocationProbe {
  explicit AllocationProbe(bool fail = false) {
    allocations = 0;
    fail_next = fail;
    tracking = true;
  }
  ~AllocationProbe() {
    tracking = false;
    fail_next = false;
  }
};

template <typename F>
std::size_t count_allocations(F&& action) {
  AllocationProbe probe;
  action();
  return allocations;
}

using S = std::uint32_t;
using Owner = kakuhen::ndarray::NDArray<int, S>;
using View = kakuhen::ndarray::NDView<int, S>;
using Slice = kakuhen::ndarray::Slice<S>;
}  // namespace

void* operator new(std::size_t size) {
  if (tracking) {
    ++allocations;
    if (std::exchange(fail_next, false)) throw std::bad_alloc();
  }
  if (void* ptr = std::malloc(size == 0 ? 1 : size)) return ptr;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
  return ::operator new(size);
}
void operator delete(void* ptr) noexcept {
  std::free(ptr);
}
void operator delete[](void* ptr) noexcept {
  std::free(ptr);
}
void operator delete(void* ptr, std::size_t) noexcept {
  std::free(ptr);
}
void operator delete[](void* ptr, std::size_t) noexcept {
  std::free(ptr);
}

TEST_CASE("NDView factories copies and transforms allocate nothing through rank four",
          "[ndview][allocation]") {
  for (S rank = 1; rank <= 4; ++rank) {
    CAPTURE(rank);
    std::vector<S> shape(rank, 2);
    std::vector<Slice> slices(rank);
    Owner owner(shape);
    owner.fill(3);
    View view, copy, transformed;
    auto count = count_allocations([&] {
      view = owner.view();
      copy = view;
      transformed = owner.reshape(shape);
      transformed = view.reshape(std::span<const S>(shape));
      transformed = owner.slice(slices);
      transformed = view.slice(std::span<const Slice>(slices));
      transformed.fill(7);
    });
    REQUIRE(count == 0);
    REQUIRE(copy.size() == owner.size());
    REQUIRE(transformed.strides().back() == 1);
  }
  Owner owner({2, 2, 2, 2});
  View view;
  auto count = count_allocations([&] {
    view = owner.reshape({4, 4});
    view = view.reshape({2, 2, 2, 2});
    view = owner.slice({{}, {}, {}, {}});
    view = view.slice({{}, {}, {}, {}});
    view = owner.diagonal(0, 1);
  });
  REQUIRE(count == 0);
  REQUIRE(view.size() == 8);
}

TEST_CASE("NDView higher ranks fall back to the heap and move without allocation",
          "[ndview][allocation]") {
  Owner owner({8, 1, 1, 1, 1, 1, 1, 1, 1});
  owner.fill(5);
  View view;
  REQUIRE(count_allocations([&] { view = owner.view(); }) > 0);
  View copied;
  REQUIRE(count_allocations([&] { copied = view; }) > 0);
  REQUIRE(copied.ndim() == 9);
  REQUIRE(copied(7, 0, 0, 0, 0, 0, 0, 0, 0) == 5);
  View moved;
  View destination;
  auto count = count_allocations([&] {
    moved = View(std::move(view));
    destination = View(std::move(moved));
  });
  REQUIRE(count == 0);
  REQUIRE(view.empty());
  REQUIRE(moved.empty());
  REQUIRE(destination(7, 0, 0, 0, 0, 0, 0, 0, 0) == 5);
}

TEST_CASE("SmallVector copy assignment keeps its buffer when growing fails",
          "[small_vector][allocation]") {
  using Vec = kakuhen::util::SmallVector<int, 2>;
  Vec target{1, 2, 3};  // heap-backed, capacity 3
  const Vec source{4, 5, 6, 7};
  bool threw = false;
  {
    AllocationProbe probe(true);
    try {
      target = source;
    } catch (const std::bad_alloc&) {
      threw = true;
    }
  }
  REQUIRE(threw);
  REQUIRE(target == Vec{1, 2, 3});
}  // the destructor must release the original buffer exactly once
