#pragma once

#include <filesystem>
#include <iostream>
#include <optional>
#include <type_traits>

namespace kakuhen::integrator {

/*!
 * @brief Configuration options for integrators.
 *
 * This struct defines a set of optional parameters that can be used to
 * configure the behavior of the integrators. All options are `std::optional`
 * to allow for default values to be used when an option is not explicitly set.
 *
 * It supports designated initializers (C++20) for easy configuration:
 * `Options{.neval = 1000, .seed = 42}`
 *
 * @tparam T The value type for integral results (e.g., double).
 * @tparam U The count type for the number of evaluations (e.g., uint64_t).
 * @tparam R The seed type for the random number generator (e.g., uint64_t).
 */
template <typename T, typename U, typename R>
struct Options {
  using value_type = T;
  using count_type = U;
  using seed_type = R;

  std::optional<count_type> neval;         //!< Number of evaluations for an integration step.
  std::optional<count_type> niter;         //!< Number of iterations for an integration run.
  std::optional<bool> adapt;               //!< Whether to run adaptation after integration.
  std::optional<bool> collect_adapt_data;  //!< Whether to collect data needed for adaptation.
  std::optional<seed_type> seed;           //!< Seed of the random number generator.
  std::optional<value_type> rel_tol;       //!< Relative precision goal for convergence.
  std::optional<value_type> abs_tol;       //!< Absolute precision goal for convergence.
  std::optional<int> verbosity;            //!< Verbosity level of output messages.
  std::optional<void*> user_data;          //!< Pointer to user-defined data (non-owning).
  std::optional<std::filesystem::path> file_path;  //!< Path for saving state/data.

  /*!
   * @brief Sets options from another Options object.
   *
   * Only non-empty optional values from the input `opts` will overwrite
   * the current object's corresponding optional values.
   *
   * @param opts The Options object to take values from.
   */
  inline void set(const Options& opts) {
    if (opts.neval) neval = *opts.neval;
    if (opts.niter) niter = *opts.niter;
    if (opts.adapt) adapt = *opts.adapt;
    if (opts.collect_adapt_data) collect_adapt_data = *opts.collect_adapt_data;
    if (opts.seed) seed = *opts.seed;
    if (opts.rel_tol) rel_tol = *opts.rel_tol;
    if (opts.abs_tol) abs_tol = *opts.abs_tol;
    if (opts.verbosity) verbosity = *opts.verbosity;
    if (opts.user_data) user_data = *opts.user_data;
    if (opts.file_path) file_path = *opts.file_path;
  }

  /*!
   * @brief Applies the current options to another Options object.
   *
   * This is a convenience method that calls `opts.set(*this)`.
   *
   * @param opts The Options object to apply values to.
   */
  inline void apply(Options& opts) const noexcept {
    opts.set(*this);
  }

  /*!
   * @brief Stream insertion operator for Options.
   *
   * Prints the set options in a readable format.
   *
   * @param os The output stream.
   * @param opts The Options object to print.
   * @return The output stream.
   */
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
    add(".collect_adapt_data", opts.collect_adapt_data);
    add(".seed", opts.seed);
    add(".rel_tol", opts.rel_tol);
    add(".abs_tol", opts.abs_tol);
    add(".verbosity", opts.verbosity);
    add(".user_data", opts.user_data);
    if (opts.file_path) {
      if (comma) os << ", ";
      os << ".file_path=\"" << opts.file_path->string() << std::string("\"");
      comma = true;
    }

    return os << "}";
  }

  /*!
   * @brief Helper for keyword-argument-like option setting.
   *
   * This struct facilitates a fluent interface for setting individual options
   * using a syntax similar to keyword arguments in other languages.
   *
   * @tparam MemberPtr A pointer to a member of the `Options` struct.
   */
  template <auto MemberPtr>
  struct OptionKey {
    template <typename V>
    struct Setter {
      V value;
      inline void apply(Options& opts) const noexcept {
        opts.*MemberPtr = std::forward<V>(value);
      }
    };
    template <typename V>
    constexpr Setter<std::decay_t<V>> operator=(V&& v) const {
      return {std::forward<V>(v)};
    }
  };

  /*!
   * @brief Provides keyword objects for setting options.
   *
   * This struct contains static `OptionKey` members for each option field,
   * allowing for syntax like `keys::neval = 1000`.
   *
   * Usage example: `integrator.integrate(func, Options::keys::neval = 1000);`
   */
  struct keys {
    static constexpr OptionKey<&Options::neval> neval{};  //!< Key for `neval` option.
    static constexpr OptionKey<&Options::niter> niter{};  //!< Key for `niter` option.
    static constexpr OptionKey<&Options::adapt> adapt{};  //!< Key for `adapt` option.
    static constexpr OptionKey<&Options::collect_adapt_data>
        collect_adapt_data{};                           //!< Key for `collect_adapt_data` option.
    static constexpr OptionKey<&Options::seed> seed{};  //!< Key for `seed` option.
    static constexpr OptionKey<&Options::rel_tol> rel_tol{};      //!< Key for `rel_tol` option.
    static constexpr OptionKey<&Options::abs_tol> abs_tol{};      //!< Key for `abs_tol` option.
    static constexpr OptionKey<&Options::verbosity> verbosity{};  //!< Key for `verbosity` option.
    static constexpr OptionKey<&Options::user_data> user_data{};  //!< Key for `user_data` option.
    static constexpr OptionKey<&Options::file_path> file_path{};  //!< Key for `file_path` option.
  };  // struct keys

};  // struct Options

}  // namespace kakuhen::integrator
