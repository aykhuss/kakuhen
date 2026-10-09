// separate executable so the allocation failures can't affect other test suites
#include "kakuhen/integrator/basin_generator.h"
#include "kakuhen/integrator/vegas_generator.h"
#include "test_support.h"
#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <new>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
thread_local long allocation_budget = -1;
thread_local bool allocation_failed = false;

struct FailAllocations {
  explicit FailAllocations(long budget) {
    allocation_budget = budget;
    allocation_failed = false;
  }
  ~FailAllocations() {
    allocation_budget = -1;
  }
};
}  // namespace

// only the plain forms are replaced; an over-aligned allocation
// (`operator new(size, align_val_t)`) would slip past the budget unseen
void* operator new(std::size_t size) {
  if (allocation_budget == 0) {
    allocation_failed = true;
    throw std::bad_alloc();
  }
  if (allocation_budget > 0) --allocation_budget;
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

using namespace kakuhen::integrator;
using test_support::TempDir;

namespace {
template <typename G>
struct Fixture : G {
  using G::env_propose;
  using G::envelope_;
  using G::G;

  void configure(bool conditional) {
    this->set_options({.frozen = true, .verbosity = 0});
    if constexpr (std::same_as<G, BasinGenerator<>>) {
      if (conditional) {
        this->nblocks_ = 1;
        for (typename G::size_type i = 1; i < this->ndim_; ++i)
          this->order_(i, 0) = i - 1;
        for (typename G::size_type i = 0; i < this->ndim_; ++i)
          std::copy_n(&this->grid_(this->order_(i, 0), this->order_(i, 1), 0, 0), this->ndiv0_,
                      &this->ordered_grid_(i, 0, 0));
      }
    }
  }
};

template <typename G>
auto make_generator(bool conditional, bool initialized) {
  auto gen = [] {
    if constexpr (std::same_as<G, BasinGenerator<>>)
      return Fixture<G>(3, 2, 2);
    else
      return Fixture<G>(3, 4);
  }();
  gen.configure(conditional);
  if (initialized) gen.initialize_envelope([](const Point<>&) { return 2.0; }, 8ULL);
  gen.set_seed(23);
  return gen;
}

std::string snapshot(const auto& gen) {
  std::ostringstream out;
  gen.write_state_stream(out);
  gen.write_rng_state_stream(out);
  return out.str();
}

std::vector<double> proposals(auto& gen) {
  Point<> point(gen.ndim());
  std::vector<double> result;
  for (unsigned i = 0; i < 16; ++i) {
    result.push_back(gen.env_propose(point));
    result.push_back(point.weight);
    result.insert(result.end(), point.x.begin(), point.x.end());
  }
  return result;
}
}  // namespace

TEMPLATE_TEST_CASE("Every failed envelope allocation preserves state and permits retry",
                   "[generator][envelope_files][allocation]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-allocation");
  const auto path = files.dir / "worker.khe";
  for (const bool conditional : {false, true}) {
    if constexpr (std::same_as<TestType, VegasGenerator<>>) {
      if (conditional) continue;
    }
    auto source = make_generator<TestType>(conditional, true);
    for (typename TestType::size_type i = 0; i < source.envelope_.size(); ++i)
      source.envelope_[i] = 1.0 + i % 5;
    source.scale_envelope(1.0);
    source.collect_envelope([](const Point<>&) { return 100.0; }, 16ULL);
    source.save_envelope(path);
    const auto collect_local = [](auto& gen) {
      gen.collect_envelope([](const Point<>&) { return 100.0; }, 8ULL);
      gen.set_seed(23);
    };
    for (const bool initialized : {false, true}) {
      CAPTURE(conditional, initialized);
      auto reference = make_generator<TestType>(conditional, initialized);
      if (initialized) collect_local(reference);
      reference.merge_envelope(path);
      const auto expected_proposals = proposals(reference);
      bool completed = false;
      // fail each allocation in turn, keeping failures armed while unwinding;
      // stop once a whole import no longer reaches the injected failure
      for (long budget = 0; budget < 256; ++budget) {
        CAPTURE(budget);
        auto target = make_generator<TestType>(conditional, initialized);
        if (initialized) collect_local(target);
        std::vector<double> previous_proposals;
        if (initialized) {
          previous_proposals = proposals(target);
          target.set_seed(23);
        }
        const auto before = snapshot(target);
        const auto volume_before = target.envelope_volume();
        bool threw = false;
        {
          FailAllocations fail(budget);
          try {
            target.merge_envelope(path);
          } catch (...) {
            threw = true;
          }
        }
        const bool failure_reached = allocation_failed;
        if (threw) {
          REQUIRE(failure_reached);
          REQUIRE(snapshot(target) == before);
          REQUIRE(target.envelope_ready() == initialized);
          REQUIRE(target.envelope_volume() == volume_before);
          if (initialized) {
            REQUIRE(proposals(target) == previous_proposals);
            target.set_seed(23);
          }
          // a failed first import must not poison the caches used by a retry
          REQUIRE_NOTHROW(target.merge_envelope(path));
        }
        REQUIRE(target.envelope_ready());
        REQUIRE(target.envelope_volume() == reference.envelope_volume());
        REQUIRE(proposals(target) == expected_proposals);
        REQUIRE_NOTHROW(target.raise_envelope([](const Point<>&) { return 1.0; }, 4ULL));
        if (!failure_reached) {
          completed = true;
          break;
        }
      }
      REQUIRE(completed);
    }
  }
}

TEMPLATE_TEST_CASE("Failed local replay allocations preserve pending observations and proposals",
                   "[generator][envelope_data][allocation]", VegasGenerator<>, BasinGenerator<>) {
  for (const bool conditional : {false, true}) {
    if constexpr (std::same_as<TestType, VegasGenerator<>>) {
      if (conditional) continue;
    }
    const auto prepare = [&](auto& gen) {
      gen.collect_envelope([](const Point<>&) { return 100.0; }, 32ULL);
      gen.set_seed(23);
    };
    auto reference = make_generator<TestType>(conditional, true);
    prepare(reference);
    reference.adapt_envelope();
    const auto expected_proposals = proposals(reference);
    bool completed = false;
    for (long budget = 0; budget < 256; ++budget) {
      CAPTURE(conditional, budget);
      auto target = make_generator<TestType>(conditional, true);
      prepare(target);
      const auto before = snapshot(target);
      const auto previous_proposals = proposals(target);
      target.set_seed(23);
      bool threw = false;
      {
        FailAllocations fail(budget);
        try {
          target.adapt_envelope();
        } catch (...) {
          threw = true;
        }
      }
      const bool failure_reached = allocation_failed;
      if (threw) {
        REQUIRE(failure_reached);
        REQUIRE(snapshot(target) == before);
        REQUIRE(target.n_envelope_records() == 32);
        REQUIRE(proposals(target) == previous_proposals);
        target.set_seed(23);
        REQUIRE_NOTHROW(target.adapt_envelope());
      }
      REQUIRE_FALSE(target.has_envelope_data());
      REQUIRE(target.abs_integral_estimate().count() == 40);
      REQUIRE(proposals(target) == expected_proposals);
      if (!failure_reached) {
        completed = true;
        break;
      }
    }
    REQUIRE(completed);
  }
}
