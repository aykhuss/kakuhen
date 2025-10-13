#pragma once

#include <filesystem>
#include <iostream>
#include <optional>

namespace kakuhen::integrator {

template <typename T, typename U, typename R>
struct Options {
  using value_type = T;
  using count_type = U;
  using seed_type = R;

  std::optional<count_type> neval;                 // number of evaluations
  std::optional<count_type> niter;                 // number of iterations
  std::optional<bool> adapt;                       // run adapt after integration
  std::optional<seed_type> seed;                   // seed of random number generator
  std::optional<value_type> rel_tol;               // relative precision goal
  std::optional<value_type> abs_tol;               // absolute precision goal
  std::optional<int> verbosity;                    // verbosity level
  std::optional<void*> user_data;                  // pointer to user data
  std::optional<std::filesystem::path> file_path;  // path for saving state/data

  inline void set(const Options& opts) {
    if (opts.neval) neval = *opts.neval;
    if (opts.niter) niter = *opts.niter;
    if (opts.adapt) adapt = *opts.adapt;
    if (opts.seed) seed = *opts.seed;
    if (opts.rel_tol) rel_tol = *opts.rel_tol;
    if (opts.abs_tol) abs_tol = *opts.abs_tol;
    if (opts.verbosity) verbosity = *opts.verbosity;
    if (opts.user_data) user_data = *opts.user_data;
  }

  inline void apply(Options& opts) const noexcept {
    opts.set(*this);
  }

  friend std::ostream& operator<<(std::ostream& os, const Options& opts) {
    os << "Options{";
    bool comma = false;

    auto add = [&](std::string_view name, const auto& opt) {
      if (opt) {
        if (comma) os << ", ";
        os << name << "=" << *opt;
        comma = true;
      }
    };

    add(".neval", opts.neval);
    add(".niter", opts.niter);
    add(".adapt", opts.adapt);
    add(".seed", opts.seed);
    add(".rel_tol", opts.rel_tol);
    add(".abs_tol", opts.abs_tol);
    add(".verbosity", opts.verbosity);
    add(".user_data", opts.user_data);

    return os << "}";
  }

  /// Option Key mechanism
  template <auto MemberPtr>
  struct OptionKey {
    template <typename V>
    struct Setter {
      V value;
      inline void apply(Options& opts) const noexcept {
        opts.*MemberPtr = value;
      }
    };
    template <typename V>
    constexpr Setter<std::decay_t<V>> operator=(V&& v) const {
      return {std::forward<V>(v)};
    }
  };

  /// keyword objects
  struct keys {
    static constexpr OptionKey<&Options::neval> neval{};
    static constexpr OptionKey<&Options::niter> niter{};
    static constexpr OptionKey<&Options::adapt> adapt{};
    static constexpr OptionKey<&Options::seed> seed{};
    static constexpr OptionKey<&Options::rel_tol> rel_tol{};
    static constexpr OptionKey<&Options::abs_tol> abs_tol{};
    static constexpr OptionKey<&Options::verbosity> verbosity{};
    static constexpr OptionKey<&Options::user_data> user_data{};
  };  // struct keys

};  // struct Options

}  // namespace kakuhen::integrator
