#pragma once
#include <cstddef>
#include <optional>

namespace kakuhen::ndarray {

//> Convenience nullopt alias, e.g., Slice(_, 5)
constexpr std::nullopt_t _ = std::nullopt;

template <typename S>
struct Slice {
  using size_type = S;

  std::optional<S> start;
  std::optional<S> stop;
  std::optional<S> step;

  //> Single index slice (e.g., Slice(3) == [3:4])
  constexpr Slice(S s) : start(s), stop(s + 1), step(1) {}

  //> Full slice (equivalent to [:])
  constexpr Slice()
      : start(std::nullopt), stop(std::nullopt), step(std::nullopt) {}

  //> Explicit slice
  constexpr Slice(std::optional<S> s, std::optional<S> e,
                  std::optional<S> st = std::nullopt)
      : start(s), stop(e), step(st) {}

  // //> Convenience nullopt alias, e.g., Slice(Slice::_, 5)
  // static inline constexpr std::nullopt_t _ = std::nullopt;

  //> Convenience factory: Slice::range(1, 5, 2)
  static constexpr Slice range(std::optional<S> s, std::optional<S> e,
                               std::optional<S> st = std::nullopt) {
    return Slice(s, e, st);
  }

  //> Convenience factory: Slice::all()
  static constexpr Slice all() {
    return Slice();
  }
};  // struct Slice

}  // namespace kakuhen::ndarray
