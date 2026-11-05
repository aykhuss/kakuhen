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
