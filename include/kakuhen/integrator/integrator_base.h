#pragma once

#include "kakuhen/integrator/integral_accumulator.h"
#include "kakuhen/integrator/integrator_feature.h"
#include "kakuhen/integrator/options.h"
#include "kakuhen/integrator/point.h"
#include "kakuhen/integrator/progress.h"
#include "kakuhen/integrator/result.h"
#include "kakuhen/util/numeric_traits.h"
#include "kakuhen/util/printer.h"
#include "kakuhen/util/scope_exit.h"
#include "kakuhen/util/serialize.h"
#include "kakuhen/util/type.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

namespace kakuhen::integrator {

/// @brief Identifiers for the different available integrators.
enum class IntegratorId : uint8_t {
  PLAIN = 0,  //!< Plain Monte Carlo integrator.
  VEGAS = 1,  //!< VEGAS adaptive integrator.
  BASIN = 2,  //!< BASIN adaptive integrator.
};

/*!
 * @brief Converts an IntegratorId to its string representation.
 * @param id The IntegratorId to convert.
 * @return A string view of the integrator name.
 */
constexpr std::string_view to_string(IntegratorId id) noexcept {
  switch (id) {
    case IntegratorId::PLAIN:
      return "Plain";
    case IntegratorId::VEGAS:
      return "Vegas";
    case IntegratorId::BASIN:
      return "Basin";
  }
  return "Unknown";
}

namespace detail {
// File format constants
constexpr std::string_view file_signature = "KAKUHEN\0";  // 8 bytes
constexpr std::size_t file_signature_size = file_signature.size();
enum class FileType : uint8_t { STATE = 0, DATA = 1 };
constexpr std::string_view suffix_state = ".khs";
constexpr std::string_view suffix_data = ".khd";
}  // namespace detail

/// @brief Default type definitions for integrator template parameters.
template <typename NT = util::num_traits_t<>>
struct IntegratorDefaults {
  using rng_type = std::mt19937_64;
  using dist_type = std::uniform_real_distribution<typename NT::value_type>;
};

/*!
 * @brief Base class for all integrators using CRTP.
 *
 * This class provides the common infrastructure for all Monte Carlo integrators
 * in the kakuhen library. It handles the main integration loop, options
 * management, result accumulation, and state serialization. The Curiously
 * Recurring Template Pattern (CRTP) is used to achieve static polymorphism,
 * allowing for compile-time dispatch to the specific integrator's
 * implementation.
 *
 * @tparam Derived The derived integrator class (e.g., Vegas, Plain).
 * @tparam NT The numeric traits for the integrator, defining value_type,
 * size_type, and count_type.
 * @tparam RNG The random number generator to use.
 * @tparam DIST The random number distribution to use.
 */
template <typename Derived, typename NT = util::num_traits_t<>,
          typename RNG = typename IntegratorDefaults<NT>::rng_type,
          typename DIST = typename IntegratorDefaults<NT>::dist_type>
class IntegratorBase {
 public:
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  using count_type = typename num_traits::count_type;
  using seed_type = typename RNG::result_type;
  using point_type = Point<num_traits>;
  using int_acc_type = IntegralAccumulator<value_type, count_type>;
  using result_type = Result<value_type, count_type>;
  using options_type = Options<value_type, count_type, seed_type>;
  using progress_event_type = ProgressEvent<value_type, count_type>;

  /*!
   * @brief Constructs an integrator base.
   *
   * @param ndim The number of dimensions for the integration.
   * @param opts Initial options for the integrator.
   */
  explicit IntegratorBase(size_type ndim = 0, const options_type& opts = {})
      : ndim_{ndim}, random_generator_{}, uniform_distribution_{0, 1}, opts_{opts} {
    // if the integrator supports adaption, set default to true
    if constexpr (detail::HasAdapt<Derived>) {
      if (!opts_.adapt) opts_.adapt = true;
      if (!opts_.frozen) opts_.frozen = false;
    }
    // set default values for options
    if (!opts_.niter) opts_.niter = 1;
    if (!opts_.verbosity) opts_.verbosity = 2;
    if (!opts_.seed) opts_.seed = 1;  // Default seed

    // Initialize RNG
    random_generator_.seed(opts_.seed.value());
  }

  /*!
   * @brief Get the unique ID of the integrator type.
   *
   * @return The compile-time `IntegratorId` of the derived class.
   */
  [[nodiscard]] static constexpr IntegratorId class_id() noexcept {
    return Derived::static_id();
  }

  /*!
   * @brief Get the unique ID of the integrator.
   *
   * @return The `IntegratorId` of the derived class.
   */
  [[nodiscard]] constexpr IntegratorId id() const noexcept {
    return class_id();
  }

  /*!
   * @brief Gets the number of dimensions for the integration.
   *
   * @return The number of dimensions.
   */
  [[nodiscard]] size_type ndim() const noexcept {
    return ndim_;
  }

  /// @name Options & Configuration
  /// @{

  /*!
   * @brief Set the persistent options for the integrator.
   *
   * These options will be used for all subsequent integration calls, unless
   * they are temporarily overridden by the options passed to the `integrate`
   * method.
   *
   * @param opts The options to set.
   * @throws std::invalid_argument if an option is incompatible with the integrator's features.
   */
  inline void set_options(const options_type& opts) {
    if constexpr (!detail::HasAdapt<Derived>) {
      if (opts.adapt && *opts.adapt) {
        throw std::invalid_argument(std::string(to_string(id())) +
                                    " does not support grid adaption");
      }
    }
    opts_.set(opts);
    if constexpr (detail::HasAdapt<Derived>) {
      if (opts_.frozen && *opts_.frozen) {
        opts_.adapt = false;
      }
    }
    if (opts.seed) random_generator_.seed(opts_.seed.value());
  }

  /*!
   * @brief Sets the seed for the random number generator.
   *
   * @param seed The seed value.
   */
  inline void set_seed(seed_type seed) noexcept {
    set_options({.seed = seed});
  }
  /*!
   * @brief Sets a new seed for the random number generator.
   *
   * If a seed has been set before, it increments the seed by one.
   */
  inline void set_seed() {
    // opts_.seed is guaranteed to have a value from construction
    set_options({.seed = opts_.seed.value() + 1});
  }
  /*!
   * @brief Gets the current seed of the random number generator.
   *
   * @return The current seed value.
   */
  [[nodiscard]] seed_type seed() const noexcept {
    return opts_.seed.value();
  }

  /*!
   * @brief Sets the user data pointer.
   *
   * @param user_data A pointer to user-defined data.
   */
  inline void set_user_data(void* user_data = nullptr) noexcept {
    set_options({.user_data = user_data});
  }
  /*!
   * @brief Gets the user data pointer.
   *
   * @return The user data pointer.
   */
  [[nodiscard]] void* user_data() const noexcept {
    return opts_.user_data.value_or(nullptr);
  }

  /// @}

  /// @name Integration
  /// @{

  /*!
   * @brief The main integration routine, using keyword-style options.
   *
   * This overload allows for setting integration options using a more
   * readable, keyword-argument-like syntax.
   *
   * Example: `integrate(func, keys::neval = 1000, keys::niter = 5)`
   *
   * @tparam I The type of the integrand function.
   * @tparam Keys A parameter pack of option setters.
   * @param integrand The function to integrate.
   * @param keys A comma-separated list of option setters.
   * @return A `result_type` object containing the result of the integration.
   */
  template <typename I, typename... Keys>
  result_type integrate(I&& integrand, const Keys&... keys) {
    options_type opts{};
    (keys.apply(opts), ...);
    return integrate(std::forward<I>(integrand), opts, nullptr);
  }

  /*!
   * @brief The main integration routine (without progress callback).
   *
   * @tparam I The type of the integrand function.
   * @param integrand The function to integrate.
   * @param opts The options for this integration call.
   * @return A `result_type` object containing the result of the integration.
   */
  template <typename I>
  result_type integrate(I&& integrand, const options_type& opts) {
    return integrate(std::forward<I>(integrand), opts, nullptr);
  }

  /*!
   * @brief The main integration routine with progress callback.
   *
   * This method performs the numerical integration of the given function. It
   * iterates a number of times, and in each iteration, it calls the
   * `integrate_impl` method of the derived integrator class. The results of
   * each iteration are accumulated, and a final result is returned.
   *
   * The progress callback is invoked at lifecycle events (START, ITER_END, END)
   * and at milestone intervals specified by `opts.progress_step`. The callback
   * can return `EventSignal::CANCEL` to stop integration early.
   *
   * @tparam I The type of the integrand function.
   * @tparam CB The type of the progress callback (or std::nullptr_t).
   * @param integrand The function to integrate. It should take a `point_type`
   * as input and return a `value_type`.
   * @param opts The options for this integration call. These are temporary
   * overrides for the persistent options of the integrator.
   * @param progress_cb The progress callback, or nullptr for no callback.
   * @return A `result_type` object containing the result of the integration.
   * @throws std::invalid_argument if required options like `neval` or `niter` are missing,
   *         or if `progress_step` is invalid.
   */
  template <typename I, typename CB>
  result_type integrate(I&& integrand, const options_type& opts, CB&& progress_cb) {
    // local lvalue reference to make it callable multiple times
    auto& integrand_ref = integrand;
    // set up local options & check settings
    options_type orig_opts = opts_;
    auto restore_orig = kakuhen::util::defer([this, orig_opts] { opts_ = orig_opts; });
    set_options(opts);
    if (!opts_.neval) {
      throw std::invalid_argument("number of evaluations (neval) not set");
    }
    if (!opts_.niter) {
      throw std::invalid_argument("number of iterations (niter) not set");
    }

    constexpr bool has_callback = is_progress_callback_v<CB>;

    result_type result;

    // Always create tracker on the stack; only populated when has_callback = true.
    // Lifetime is scoped to this call, so the result pointer below never dangles.
    ProgressTracker tracker{};

    if constexpr (has_callback) {
      double progress_step = opts_.progress_step.value_or(DEFAULT_PROGRESS_STEP);
      if (progress_step <= 0.0 || progress_step > 1.0) {
        throw std::invalid_argument("progress_step must be > 0 and <= 1");
      }
      tracker.signal = EventSignal::NONE;
      tracker.niter = *opts_.niter;
      tracker.current_iter = 0;
      tracker.neval = *opts_.neval;
      tracker.current_eval = 0;
      tracker.step_milestone_eval =
          std::max(count_type{1},
                   static_cast<count_type>(static_cast<double>(tracker.neval) * progress_step));
      tracker.next_milestone_eval = 0;
      tracker.result = &result;
      tracker.time_start = std::chrono::steady_clock::now();
      tracker.time_iter = tracker.time_start;

      // Fire START event
      EventSignal sig = fire_progress_event(tracker, progress_cb, ProgressEventKind::START);
      if (has_signal(sig, EventSignal::CANCEL)) {
        fire_progress_event(tracker, progress_cb, ProgressEventKind::END);
        return result;
      }
    }

    // call the integration implementation for each iteration & accumulate
    for (count_type iter = 0; iter < *opts_.niter; ++iter) {
      // track start time and fire ITER_START event
      const auto iter_start_time = std::chrono::steady_clock::now();
      if constexpr (has_callback) {
        if (tracker.is_cancelled()) break;
        tracker.current_iter = iter;
        tracker.current_eval = 0;
        tracker.time_iter = iter_start_time;
        tracker.next_milestone_eval = tracker.step_milestone_eval;
        EventSignal sig_start =
            fire_progress_event(tracker, progress_cb, ProgressEventKind::ITER_START);
        tracker.signal |= sig_start;
        if (tracker.is_cancelled()) break;
      }

      int_acc_type res_it =
          derived().integrate_impl(integrand_ref, *opts_.neval, tracker, progress_cb);
      result.accumulate(res_it);  // always accumulate, including partial cancelled iterations

      // compute elapsed time and fire ITER_END event
      const auto iter_end_time = std::chrono::steady_clock::now();
      const std::chrono::duration<double> elapsed = iter_end_time - iter_start_time;
      if constexpr (has_callback) {
        if (tracker.is_cancelled()) {
          // Discard partial adaptive accumulation so the next integrate() call starts clean.
          if constexpr (detail::HasAdapt<Derived>) {
            if (opts_.adapt && *opts_.adapt) derived().clear_data();
          }
          break;
        }
        tracker.current_eval = tracker.neval;  // mark iteration as fully completed
        EventSignal sig = fire_progress_event(tracker, progress_cb, ProgressEventKind::ITER_END);
        if (has_signal(sig, EventSignal::CANCEL)) {
          tracker.signal |= sig;
          if constexpr (detail::HasAdapt<Derived>) {
            if (opts_.adapt && *opts_.adapt) derived().clear_data();
          }
          break;
        }
      }

      if (opts_.verbosity && *opts_.verbosity > 0) {
        print_iteration_summary(iter + 1, *opts_.niter, res_it, result, elapsed.count());
      }

      // // check for convergence
      // bool converged = false;
      // if (opts_.rel_tol) {
      //   converged =
      //       converged || res.error() <= std::abs(res.value()) *
      //       *opts_.rel_tol;
      // }
      // if (opts_.abs_tol) {
      //   converged = converged || res.error() <= *opts_.abs_tol;
      // }
      // if (converged) {
      //   if (opts_.verbosity && *opts_.verbosity > 0) {
      //     std::cout << "Converged.\n";
      //   }
      //   break;
      // }

      // adapt the grid if requested
      if constexpr (detail::HasAdapt<Derived>) {
        if (opts_.adapt && *opts_.adapt) {
          derived().adapt();
        }
      }

      // save state/data if requested
      if constexpr (detail::HasStateStream<Derived> && detail::HasPrefix<Derived>) {
        if (opts_.file_path) {
          derived().save();
        }
      }

    }  // for iter

    // Fire END event (always, even on cancel/exception)
    if constexpr (has_callback) {
      fire_progress_event(tracker, progress_cb, ProgressEventKind::END);
    }

    return result;
  }

  /// @}

  /// @name Output & Serialization
  /// @{

  /*!
   * @brief Prints the state of the integrator using a specified printer.
   *
   * This method serializes the integrator's configuration and state
   * (if supported) to a structured format (e.g., JSON) using the provided
   * printer object.
   *
   * @tparam P The type of the printer, which must conform to the `PrinterBase` interface.
   * @param prt The printer object to use for output.
   */
  template <typename P>
  void print(P& prt) const {
    using C = kakuhen::util::printer::Context;
    using namespace kakuhen::util::type;
    prt.reset();
    prt.template begin<C::OBJECT>();
    {
      prt.print_one("name", to_string(id()));
      prt.print_one("id", static_cast<std::underlying_type_t<IntegratorId>>(id()));
      prt.print_one("value_type", get_type_name<value_type>());
      prt.print_one("size_type", get_type_name<size_type>());
      prt.print_one("count_type", get_type_name<count_type>());
      prt.print_one("ndim", ndim_);
      if constexpr (detail::HasStateStream<Derived>) {
        derived().print_state(prt);
      }
    }
    prt.template end<C::OBJECT>(true);
  }

  // save state of the integrator to a file
  /*!
   * @brief Save the state of the integrator to a file.
   *
   * This method serializes the internal state of the integrator to the
   * specified file. This allows the integration to be resumed later.
   *
   * @note Available only if the derived type models `detail::HasStateStream`.
   *
   * @param filepath The path to the file where the state should be saved.
   */
  template <typename D = Derived>
  void save(const std::filesystem::path& filepath) const
    requires detail::HasStateStream<D>
  {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
      throw std::ios_base::failure("Failed to open state file: " + filepath.string());
    }
    write_header(ofs, detail::FileType::STATE);
    derived().write_state_stream(ofs);
    if (!ofs) {
      throw std::ios_base::failure("Error writing state file: " + filepath.string());
    }
  }

  /**
   * @brief Save the state of the integrator to the default state file.
   *
   * @return The path to the saved state file.
   */
  template <typename D = Derived>
  std::filesystem::path save() const
    requires detail::HasStateStream<D>
  {
    std::filesystem::path fstate = file_state();
    save(fstate);
    return fstate;
  }

  // load state of the integrator from a file
  /*!
   * @brief Load the state of the integrator from a file.
   *
   * This method deserializes the internal state of the integrator from the
   * specified file. This allows the integration to be resumed from a previous
   * state.
   *
   * @note Available only if the derived type models `detail::HasStateStream`.
   *
   * @param filepath The path to the file from which the state should be
   * loaded.
   */
  template <typename D = Derived>
  void load(const std::filesystem::path& filepath)
    requires detail::HasStateStream<D>
  {
    std::error_code ec;
    if (std::filesystem::exists(filepath, ec)) {
      if (ec) {
        throw std::system_error(ec, "Failed to check if file exists");
      }
      std::ifstream ifs(filepath, std::ios::binary);
      if (!ifs.is_open()) {
        throw std::ios_base::failure("Failed to open state file: " + filepath.string());
      }
      read_header(ifs, detail::FileType::STATE);
      derived().read_state_stream(ifs);
      if (!ifs) {
        throw std::ios_base::failure("Error reading state file: " + filepath.string());
      }
    } else {
      print_info_message("state",
                         "state file \"" + filepath.string() + "\" not found; skipping load");
    }
  }

  /**
   * @brief Load the state of the integrator from the default state file.
   *
   * @return The path to the loaded state file.
   */
  template <typename D = Derived>
  std::filesystem::path load()
    requires detail::HasStateStream<D>
  {
    std::filesystem::path fstate = file_state();
    load(fstate);
    return fstate;
  }

  /*!
   * @brief Save accumulated data of the integrator to a file.
   *
   * This method is for integrators that support data accumulation. It serializes
   * the accumulated sample data to the specified file.
   *
   * @note Available only if the derived type models `detail::HasDataStream`.
   *
   * @param filepath The path to the file where the data should be saved.
   */
  template <typename D = Derived>
  void save_data(const std::filesystem::path& filepath) const
    requires detail::HasDataStream<D>
  {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
      throw std::ios_base::failure("Failed to open data file: " + filepath.string());
    }
    write_header(ofs, detail::FileType::DATA);
    derived().write_data_stream(ofs);
    if (!ofs) {
      throw std::ios_base::failure("Error writing data file: " + filepath.string());
    }
  }

  /**
   * @brief Save accumulated data of the integrator to the default data file.
   *
   * @return The path to the saved data file.
   */
  template <typename D = Derived>
  std::filesystem::path save_data() const
    requires detail::HasDataStream<D>
  {
    std::filesystem::path fdata = file_data();
    save_data(fdata);
    return fdata;
  }

  /*!
   * @brief Append accumulated data from a file to the integrator.
   *
   * This method deserializes accumulated sample data from a file and adds it
   * to the integrator's internal data accumulator, allowing for the combination
   * of data from multiple independent runs.
   *
   * @note Available only if the derived type models `detail::HasDataStream`.
   *
   * @param filepath The path to the file from which to append the data.
   */
  template <typename D = Derived>
  void append_data(const std::filesystem::path& filepath)
    requires detail::HasDataStream<D>
  {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::ios_base::failure("Failed to open data file: " + filepath.string());
    }
    read_header(ifs, detail::FileType::DATA);
    derived().accumulate_data_stream(ifs);
    if (!ifs) {
      throw std::ios_base::failure("Error reading data file: " + filepath.string());
    }
  }

  /**
   * @brief Append accumulated data from the default data file to the integrator.
   *
   * @return The path to the appended data file.
   */
  template <typename D = Derived>
  std::filesystem::path append_data()
    requires detail::HasDataStream<D>
  {
    std::filesystem::path fdata = file_data();
    append_data(fdata);
    return fdata;
  }

  /// @brief Writes the state of the random number generator to a stream.
  void write_rng_state_stream(std::ostream& out) const {
    out << random_generator_;
  }
  /// @brief Reads the state of the random number generator from a stream.
  void read_rng_state_stream(std::istream& in) {
    in >> random_generator_;
  }
  /// @brief Saves the random number generator state to a file.
  void save_rng_state(const std::filesystem::path& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    write_rng_state_stream(ofs);
  }
  /// @brief Loads the random number generator state from a file.
  void load_rng_state(const std::filesystem::path& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::ios_base::failure("Failed to open RNG state file: " + filepath.string());
    }
    read_rng_state_stream(ifs);
    if (!ifs) {
      throw std::ios_base::failure("Error reading RNG state file: " + filepath.string());
    }
  }

  /// @}

 protected:
  size_type ndim_;             //!< The number of dimensions for the integration.
  RNG random_generator_;       //!< The random number generator.
  DIST uniform_distribution_;  //!< The uniform random number distribution.
  options_type opts_;          //!< The persistent options for the integrator.

  /*!
   * @brief Progress tracking state passed through a single integrate() call.
   *
   * Created as a local variable inside integrate() and passed by reference to
   * integrate_impl(). This scope-bound lifetime guarantees the `result` pointer
   * is never dangling. Derived classes read/write this struct only when a
   * real callback is present (guarded by `if constexpr`).
   */
  struct ProgressTracker {
    EventSignal signal{EventSignal::NONE};
    count_type niter{0};                 ///< Total number of iterations.
    count_type current_iter{0};          ///< Current iteration (0-indexed).
    count_type neval{0};                 ///< Evaluations per iteration (denominator for fraction).
    count_type current_eval{0};          ///< Evaluations completed in current iteration.
    count_type step_milestone_eval{1};   ///< Evaluations between milestones (within one iteration).
    count_type next_milestone_eval{0};   ///< Next per-iteration milestone trigger.
    const result_type* result{nullptr};  ///< Non-owning pointer to the cross-iteration result
                                         ///< (valid only during integrate()).
    std::chrono::steady_clock::time_point time_start{};
    std::chrono::steady_clock::time_point time_iter{};

    /// @brief Check if integration has been cancelled.
    [[nodiscard]] bool is_cancelled() const noexcept {
      return has_signal(signal, EventSignal::CANCEL);
    }
  };

  /*!
   * @brief Fire a progress event to the callback.
   *
   * Static to emphasise that it has no side-effects on the integrator instance;
   * all mutable state lives in `tracker`. Exceptions thrown by the callback
   * (including from an END event) are caught and converted to EXCEPTION|CANCEL
   * so that integration always terminates cleanly.
   *
   * @tparam Cb The callback type (must not be std::nullptr_t).
   * @param tracker Progress tracking state for this integrate() call.
   * @param cb The callback to invoke.
   * @param kind The type of progress event.
   * @param evals Current evaluation count.
   * @return The signal returned by the callback.
   */
  template <typename Cb>
  static EventSignal fire_progress_event(ProgressTracker& tracker, Cb&& cb,
                                         ProgressEventKind kind) {
    if constexpr (!is_progress_callback_v<Cb>) {
      return EventSignal::NONE;
    } else {
      value_type val{0};
      value_type err{0};
      if (tracker.result && tracker.result->size() > 0) {
        val = tracker.result->value();
        err = tracker.result->error();
      }

      auto now = std::chrono::steady_clock::now();
      double elapsed_start = std::chrono::duration<double>(now - tracker.time_start).count();
      double elapsed_iter = std::chrono::duration<double>(now - tracker.time_iter).count();
      // Fraction is within the current iteration so each iteration's bar sweeps 0→1.
      // current_eval is 1-indexed (number of completed evaluations), so no +1 needed.
      double fraction = tracker.neval > 0
                            ? std::min(1.0, static_cast<double>(tracker.current_eval) /
                                                static_cast<double>(tracker.neval))
                            : 0.0;

      progress_event_type event{.kind = kind,
                                .niter = tracker.niter,
                                .current_iter = tracker.current_iter,
                                .neval = tracker.neval,
                                .current_eval = tracker.current_eval,
                                .value = val,
                                .error = err,
                                .fraction = fraction,
                                .elapsed_start = elapsed_start,
                                .elapsed_iter = elapsed_iter};

      try {
        return cb(event);
      } catch (...) {
        tracker.signal |= EventSignal::EXCEPTION | EventSignal::CANCEL;
        return EventSignal::EXCEPTION | EventSignal::CANCEL;
      }
    }
  }

  /*!
   * @brief Fire an EVAL_MILESTONE event if the current sample crosses the next threshold.
   *
   * Encapsulates the per-sample progress check that would otherwise be duplicated
   * in every derived integrator's inner loop. When `ProgressCb` is `std::nullptr_t`
   * the entire body compiles away and the function unconditionally returns `false`.
   *
   * @tparam ProgressCb The callback type (or std::nullptr_t for no callback).
   * @param tracker Progress tracking state for this integrate() call.
   * @param progress_cb The callback to invoke on a milestone.
   * @param i Zero-based index of the sample just evaluated.
   * @return `true` if integration should be cancelled (break out of the eval loop).
   */
  template <typename ProgressCb>
  [[nodiscard]] static bool check_eval_milestone(ProgressTracker& tracker,
                                                 ProgressCb&& progress_cb,
                                                 count_type i) {
    if constexpr (!is_progress_callback_v<ProgressCb>) {
      return false;
    } else {
      if ((i + 1) >= tracker.next_milestone_eval) {
        tracker.current_eval = i + 1;
        EventSignal sig =
            fire_progress_event(tracker, progress_cb, ProgressEventKind::EVAL_MILESTONE);
        tracker.signal |= sig;
        tracker.next_milestone_eval += tracker.step_milestone_eval;
      }
      return tracker.is_cancelled();
    }
  }

  /*!
   * @brief Generates a random number in the range [0, 1).
   *
   * @return A random number of `value_type`.
   */
  [[nodiscard]] inline value_type ran() noexcept {
    return uniform_distribution_(random_generator_);
  }

  /*!
   * @brief Provides access to the derived class instance.
   *
   * This is part of the CRTP pattern.
   *
   * @return A reference to the derived class.
   */
  inline Derived& derived() {
    return static_cast<Derived&>(*this);
  }

  /*!
   * @brief Provides const access to the derived class instance.
   *
   * This is part of the CRTP pattern.
   *
   * @return A const reference to the derived class.
   */
  inline const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }

 private:
  static std::string fmt_scientific(long double value, int precision = 6) {
    std::ostringstream out;
    out << std::scientific << std::setprecision(precision) << value;
    return out.str();
  }

  static std::string fmt_fixed(double value, int precision = 3) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
  }

  void print_info_message(std::string_view channel, const std::string& message) const {
    const std::string name{to_string(id())};
    std::cout << "[" << name << ":" << channel << "] " << message << "\n";
  }

  void print_iteration_summary(count_type iter, count_type niter, const int_acc_type& res_it,
                               const result_type& result, double elapsed_seconds) const {
    const int verbosity = opts_.verbosity.value_or(0);
    if (verbosity <= 0) return;

    const std::string name{to_string(id())};
    if (verbosity == 1) {
      std::cout << "[" << name << ": iter " << iter << "/" << niter << "] "
                << "I_it=" << fmt_scientific(static_cast<long double>(res_it.value())) << " +/- "
                << fmt_scientific(static_cast<long double>(res_it.error())) << " | "
                << "I_acc=" << fmt_scientific(static_cast<long double>(result.value())) << " +/- "
                << fmt_scientific(static_cast<long double>(result.error())) << " | "
                << "chi2/dof=" << result.chi2dof() << " | "
                << "t=" << fmt_fixed(elapsed_seconds) << "s\n";
      return;
    }

    std::cout << "\n=== Integration Report =====================================\n";
    std::cout << "integrator    : " << name << "\n";
    std::cout << "iteration     : " << iter << " / " << niter << "\n";
    std::cout << "integral(iter): " << fmt_scientific(static_cast<long double>(res_it.value()))
              << " +/- " << fmt_scientific(static_cast<long double>(res_it.error())) << "\n";
    std::cout << "samples(it)   : " << res_it.count() << "\n";
    std::cout << "integral(acc.): " << fmt_scientific(static_cast<long double>(result.value()))
              << " +/- " << fmt_scientific(static_cast<long double>(result.error())) << "\n";
    std::cout << "samples(acc)  : " << result.count() << "\n";
    std::cout << "chi2/dof      : " << result.chi2dof() << "\n";
    std::cout << "time          : " << fmt_fixed(elapsed_seconds) << " s\n";
    std::cout << "============================================================\n";
  }

  template <typename D = Derived>
  [[nodiscard]] inline std::filesystem::path file_state() const noexcept
    requires detail::HasPrefix<D>
  {
    std::filesystem::path fstate = derived().prefix() + std::string(detail::suffix_state);
    if (opts_.file_path) {
      fstate = *opts_.file_path;
      fstate.replace_extension(detail::suffix_state);
    }
    return fstate;
  }

  template <typename D = Derived>
  [[nodiscard]] inline std::filesystem::path file_data() const noexcept
    requires detail::HasPrefix<D>
  {
    std::string seed_suffix = ".s" + std::to_string(opts_.seed.value_or(0));
    std::filesystem::path fdata =
        derived().prefix(true) + seed_suffix + std::string(detail::suffix_data);
    if (opts_.file_path) {
      fdata = *opts_.file_path;
      fdata.replace_extension(seed_suffix + std::string(detail::suffix_data));
    }
    return fdata;
  }

  void write_header(std::ostream& out, detail::FileType ftype) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    write_bytes(out, detail::file_signature.data(), detail::file_signature_size);
    serialize_one<IntegratorId>(out, id());
    serialize_one<detail::FileType>(out, ftype);
    int16_t T_tos = get_type_or_size<value_type>();
    int16_t S_tos = get_type_or_size<size_type>();
    int16_t U_tos = get_type_or_size<count_type>();
    serialize_one<int16_t>(out, T_tos);
    serialize_one<int16_t>(out, S_tos);
    serialize_one<int16_t>(out, U_tos);
  }

  void read_header(std::istream& in, detail::FileType expected_ftype) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    // check the file signature
    std::array<char, detail::file_signature_size> buf{};
    read_bytes(in, buf.data(), detail::file_signature_size);
    if (std::string_view(buf.data(), buf.size()) != detail::file_signature) {
      throw std::runtime_error("Invalid kakuhen file signature");
    }
    // integrator id check
    IntegratorId id_chk;
    deserialize_one<IntegratorId>(in, id_chk);
    if (id_chk != id()) {
      throw std::runtime_error("Integrator id mismatch");
    }
    // file type check
    detail::FileType ftype_chk;
    deserialize_one<detail::FileType>(in, ftype_chk);
    if (ftype_chk != expected_ftype) {
      throw std::runtime_error("File type mismatch " +
                               std::to_string(static_cast<uint8_t>(ftype_chk)) +
                               " != " + std::to_string(static_cast<uint8_t>(expected_ftype)));
    }
    // type checks
    int16_t T_tos;
    deserialize_one<int16_t>(in, T_tos);
    if (T_tos != get_type_or_size<value_type>()) {
      throw std::runtime_error("type or size mismatch for typename T");
    }
    int16_t S_tos;
    deserialize_one<int16_t>(in, S_tos);
    if (S_tos != get_type_or_size<size_type>()) {
      throw std::runtime_error("type or size mismatch for typename S");
    }
    int16_t U_tos;
    deserialize_one<int16_t>(in, U_tos);
    if (U_tos != get_type_or_size<count_type>()) {
      throw std::runtime_error("type or size mismatch for typename U");
    }
  }
};

/// @name Header Utilities
/// @{

/// @brief Header structure returned by `parse_header` containing integrator metadata.
struct IntegratorHeader {
  IntegratorId id;                            //!< The unique ID of the integrator.
  kakuhen::util::type::TypeId value_type_id;  //!< Type ID for the integration value type.
  kakuhen::util::type::TypeId size_type_id;   //!< Type ID for the size type.
  kakuhen::util::type::TypeId count_type_id;  //!< Type ID for the count type.
};

/*!
 * @brief Parses an integrator header from an input stream.
 *
 * This function reads a kakuhen file header from a stream and
 * returns an `IntegratorHeader` struct containing the integrator's ID and
 * type information. It is independent of the specific integrator type.
 *
 * @param in The input stream to read from.
 * @return An `IntegratorHeader` struct.
 * @throws std::runtime_error if the file signature is invalid.
 */
[[nodiscard]] inline IntegratorHeader parse_header(std::istream& in) {
  using namespace kakuhen::util::serialize;
  using namespace kakuhen::util::type;

  IntegratorHeader ret;

  // check the file signature
  std::array<char, detail::file_signature_size> buf{};
  read_bytes(in, buf.data(), detail::file_signature_size);
  if (std::string_view(buf.data(), buf.size()) != detail::file_signature) {
    throw std::runtime_error("Invalid kakuhen file signature");
  }

  deserialize_one<IntegratorId>(in, ret.id);

  // file type: skipped in external API
  detail::FileType ftype_chk;
  deserialize_one<detail::FileType>(in, ftype_chk);

  // types: read as int16_t (as written by write_header) then cast
  int16_t val;
  deserialize_one<int16_t>(in, val);
  ret.value_type_id = static_cast<TypeId>(val);
  deserialize_one<int16_t>(in, val);
  ret.size_type_id = static_cast<TypeId>(val);
  deserialize_one<int16_t>(in, val);
  ret.count_type_id = static_cast<TypeId>(val);

  return ret;
}

/*!
 * @brief Parses an integrator header from a file.
 *
 * This is a convenience wrapper around the stream version of `parse_header`.
 *
 * @param filepath The path to the file.
 * @return An `IntegratorHeader` struct.
 * @throws std::ios_base::failure if the file cannot be opened.
 */
[[nodiscard]] inline IntegratorHeader parse_header(const std::filesystem::path& filepath) {
  std::error_code ec;
  if (std::filesystem::exists(filepath, ec)) {
    if (ec) {
      throw std::system_error(ec, "Failed to check if file exists: " + filepath.string());
    }
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::ios_base::failure("Failed to open kakuhen file: " + filepath.string());
    }
    return parse_header(ifs);
  } else {
    throw std::ios_base::failure("Failed to open kakuhen file: " + filepath.string());
  }
}

/// @}

}  // namespace kakuhen::integrator
