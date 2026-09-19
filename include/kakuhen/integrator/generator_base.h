#pragma once

#include "kakuhen/integrator/integral_accumulator.h"
#include "kakuhen/integrator/integrator_base.h"
#include "kakuhen/integrator/point.h"
#include "kakuhen/integrator/progress.h"
#include "kakuhen/ndarray/ndarray.h"
#include "kakuhen/util/hash.h"
#include "kakuhen/util/math.h"
#include "kakuhen/util/numeric_traits.h"
#include "kakuhen/util/progress_bar.h"
#include "kakuhen/util/serialize.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <istream>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace kakuhen::integrator {

/// @brief How a generation run ended. Ordered by severity so that merging keeps the worst.
enum class GenerationStatus : uint8_t {
  COMPLETED = 0,  //!< All trials were used up.
  STOPPED = 1,    //!< The event callback returned `EventSignal::STOP`.
};

/*!
 * @brief Converts a GenerationStatus to its string representation.
 * @param status The GenerationStatus to convert.
 * @return A string view of the status name.
 */
constexpr std::string_view to_string(GenerationStatus status) noexcept {
  switch (status) {
    case GenerationStatus::COMPLETED:
      return "completed";
    case GenerationStatus::STOPPED:
      return "stopped";
  }
  return "unknown";
}

/*!
 * @brief Statistics of a generation run.
 *
 * Events are not normalized: they carry weight `+-1`, or `+-|f|/R` for the
 * rare overweight events. The normalization is `volume() / n_trials()` and
 * both factors are stored here. The accumulator collects the signed weight of
 * every trial (`0` for rejected trials), so that `value() = volume() * <w>`
 * is an estimate of the signed integral. The estimate is unbiased for a
 * completed run.
 *
 * Results of independent runs with the same envelope can be merged with
 * `accumulate()`.
 *
 * @tparam T The value type for the integral results (e.g., double).
 * @tparam U The count type for the number of trials/events (e.g., uint64_t).
 */
template <typename T, typename U>
struct GenerationResult {
  using value_type = T;
  using count_type = U;
  using int_acc_type = IntegralAccumulator<T, U>;

  int_acc_type acc_{};        //!< Signed weight of each trial (0 for rejected trials).
  U n_events_ = 0;            //!< Number of accepted events (incl. overweights).
  U n_overweight_ = 0;        //!< Number of accepted events above the envelope.
  U n_negative_ = 0;          //!< Number of accepted events with negative weight.
  U n_nonfinite_ = 0;         //!< Number of trials with a non-finite integrand (set to zero).
  T max_overweight_ = T(0);   //!< Largest overweight factor |f|/R (0 if none).
  T envelope_volume_ = T(0);  //!< Volume V of the envelope used for the generation.
  GenerationStatus status_ = GenerationStatus::COMPLETED;  //!< How the run ended.

  /// @name Queries
  /// @{

  /// @brief Total number of trials, accepted or not.
  [[nodiscard]] inline U n_trials() const noexcept {
    return acc_.count();
  }
  /// @brief Number of accepted events (including overweights).
  [[nodiscard]] inline U n_events() const noexcept {
    return n_events_;
  }
  /// @brief Number of rejected trials.
  [[nodiscard]] inline U n_rejected() const noexcept {
    return n_trials() - n_events_;
  }
  /// @brief Number of accepted events above the envelope (weight > 1).
  [[nodiscard]] inline U n_overweight() const noexcept {
    return n_overweight_;
  }
  /// @brief Number of accepted events with a negative weight.
  [[nodiscard]] inline U n_negative() const noexcept {
    return n_negative_;
  }
  /// @brief Number of trials with a non-finite integrand (set to zero).
  [[nodiscard]] inline U n_nonfinite() const noexcept {
    return n_nonfinite_;
  }
  /// @brief Largest overweight factor |f|/R (0 if none).
  [[nodiscard]] inline T max_overweight() const noexcept {
    return max_overweight_;
  }
  /// @brief The envelope volume V. Events are normalized with `volume() / n_trials()`.
  [[nodiscard]] inline T volume() const noexcept {
    return envelope_volume_;
  }
  /// @brief How the run ended. A merged result is only `COMPLETED` if all runs completed.
  [[nodiscard]] inline GenerationStatus status() const noexcept {
    return status_;
  }

  /// @brief Unweighting efficiency: accepted events per trial.
  [[nodiscard]] inline T efficiency() const noexcept {
    return n_trials() > U(0) ? T(n_events_) / T(n_trials()) : T(0);
  }
  /// @brief Fraction of accepted events with negative weight.
  [[nodiscard]] inline T negative_fraction() const noexcept {
    return n_events_ > U(0) ? T(n_negative_) / T(n_events_) : T(0);
  }

  /*!
   * @brief Estimate of the signed integral from this generation run.
   *
   * Computed as `V * <w>` over all trials. The estimate is unbiased for
   * completed runs (overweights included).
   *
   * @return The estimate of the signed integral.
   * @throws std::runtime_error if no trials have been accumulated.
   */
  [[nodiscard]] T value() const {
    if (n_trials() == U(0)) throw std::runtime_error("GenerationResult: no trials accumulated");
    return envelope_volume_ * acc_.value();
  }
  /*!
   * @brief Error (standard deviation) of `value()`.
   * @throws std::runtime_error if no trials have been accumulated.
   */
  [[nodiscard]] T error() const {
    if (n_trials() == U(0)) throw std::runtime_error("GenerationResult: no trials accumulated");
    return envelope_volume_ * acc_.error();
  }

  /*!
   * @brief The normalization per event `volume() / n_trials()`.
   *
   * Multiply the event weights by this factor to get a properly normalized
   * sample (e.g. to fill histograms). If the events of several runs are
   * combined, use the normalization of the merged result. Stopped runs can
   * have a biased normalization, so better use completed runs if the absolute
   * normalization matters.
   *
   * @throws std::runtime_error if no trials have been accumulated.
   */
  [[nodiscard]] T normalization() const {
    if (n_trials() == U(0)) throw std::runtime_error("GenerationResult: no trials accumulated");
    return envelope_volume_ / T(n_trials());
  }

  /// @}

  /*!
   * @brief Merge the statistics of another run that used the same envelope.
   *
   * @param other The result to merge in.
   * @throws std::invalid_argument if both results are non-empty but have
   *         different envelope volumes. The result is left unchanged.
   */
  void accumulate(const GenerationResult<T, U>& other) {
    // validate before touching any state (everything below is non-throwing)
    if (n_trials() > U(0) && other.n_trials() > U(0) &&
        envelope_volume_ != other.envelope_volume_) {
      throw std::invalid_argument(
          "GenerationResult: cannot merge runs generated against different envelopes");
    }
    // an empty result still carries a status (status ordered by severity)
    status_ = static_cast<GenerationStatus>(
        util::math::max(static_cast<uint8_t>(status_), static_cast<uint8_t>(other.status_)));
    if (other.n_trials() == U(0)) return;
    if (n_trials() == U(0)) envelope_volume_ = other.envelope_volume_;
    acc_.accumulate(other.acc_);
    n_events_ += other.n_events_;
    n_overweight_ += other.n_overweight_;
    n_negative_ += other.n_negative_;
    n_nonfinite_ += other.n_nonfinite_;
    max_overweight_ = util::math::max(max_overweight_, other.max_overweight_);
  }

};  // struct GenerationResult

/*!
 * @brief Result of a single `raise_envelope` pass.
 *
 * Holds the running estimate of the absolute integral `A = int |f| du` over
 * all passes so far, the number of envelope violations in this pass, and the
 * volume `V` of the sealed envelope. The violation count should go down over
 * repeated passes. The predicted unweighting efficiency is
 * `efficiency() = A / V`.
 *
 * @tparam T The value type for the integral results (e.g., double).
 * @tparam U The count type for the number of evaluations (e.g., uint64_t).
 */
template <typename T, typename U>
struct EnvelopeResult {
  using value_type = T;
  using count_type = U;
  using int_acc_type = IntegralAccumulator<T, U>;

  int_acc_type abs_acc_{};    //!< Running estimate of the absolute integral A.
  U n_violations_ = 0;        //!< Envelope violations in this pass.
  T envelope_volume_ = T(0);  //!< Envelope volume V after sealing.
  U n_nonfinite_ = 0;         //!< Samples with a non-finite integrand in this pass (set to zero).

  /// @name Queries
  /// @{

  /// @brief Estimate of the absolute integral A = int |f| du.
  [[nodiscard]] inline T abs_integral() const noexcept {
    return abs_acc_.value();
  }
  /// @brief Error (standard deviation) of `abs_integral()`.
  [[nodiscard]] inline T abs_error() const noexcept {
    return abs_acc_.error();
  }
  /// @brief Number of samples in the A estimate.
  [[nodiscard]] inline U count() const noexcept {
    return abs_acc_.count();
  }
  /// @brief Envelope violations in this pass.
  [[nodiscard]] inline U n_violations() const noexcept {
    return n_violations_;
  }
  /// @brief Samples with a non-finite integrand in this pass (set to zero).
  [[nodiscard]] inline U n_nonfinite() const noexcept {
    return n_nonfinite_;
  }
  /// @brief The volume V of the sealed envelope.
  [[nodiscard]] inline T volume() const noexcept {
    return envelope_volume_;
  }
  /// @brief Predicted unweighting efficiency A / V.
  [[nodiscard]] inline T efficiency() const noexcept {
    return abs_integral() / envelope_volume_;
  }

  /// @}

};  // struct EnvelopeResult

namespace detail {

/// @brief Add `weight` to an inclusive CDF sum.
/// @throws std::runtime_error unless the result is finite and strictly increasing.
template <typename T>
T cdf_step(T sum, T weight) {
  const T next = sum + weight;
  if (!(next > sum) || !std::isfinite(next)) [[unlikely]]
    throw std::runtime_error("envelope: weights must form a finite strictly increasing CDF");
  return next;
}

/// @brief A cell drawn from an inclusive CDF.
template <typename T, typename S>
struct CDFDraw {
  S cell;      //!< The selected cell.
  T fraction;  //!< Position within the CDF segment of the cell, in [0, 1).
  T width;     //!< The CDF mass of the cell `cdf[cell] - cdf[cell - 1]`.
};

/// @brief Index of the first entry of the sorted `cdf` that is greater than
///        `target` (as `std::upper_bound`).
/// The search is branchless: the targets are random, so the branches of
/// `std::upper_bound` would be mispredicted about half of the time.
template <typename T>
std::size_t upper_bound_branchless(std::span<const T> cdf, T target) {
  const T* base = cdf.data();
  std::size_t n = cdf.size();
  while (n > 1) {
    const std::size_t half = n / 2;
    base = base[half - 1] <= target ? base + half : base;
    n -= half;
  }
  return static_cast<std::size_t>(base - cdf.data()) + (*base <= target ? 1 : 0);
}

/// @brief Invert an inclusive CDF at `u` in [0, 1].
template <typename T, typename S>
CDFDraw<T, S> sample_cdf(std::span<const T> cdf, T u) {
  T target = u * cdf.back();
  if (target >= cdf.back()) [[unlikely]]
    target = std::nextafter(cdf.back(), T(0));
  const auto cell = static_cast<S>(upper_bound_branchless(cdf, target));
  const T low = cell == 0 ? T(0) : cdf[cell - 1];
  const T width = cdf[cell] - low;
  return {cell, (target - low) / width, width};
}

}  // namespace detail

/*!
 * @brief CRTP base for event generation on a frozen grid.
 *
 * Takes care of the envelope and the bookkeeping during generation. The usual
 * steps are:
 *
 * 1. `initialize_envelope(...)`: seed a flat envelope, either with a given
 *    value (e.g. `|I|` from a production run) or from a new run that
 *    estimates the absolute integral;
 * 2. `optimize_envelope(...)`: raising passes until the violation rate
 *    is small enough (a single pass is `raise_envelope(...)`);
 * 3. `generate_trials(...)`: hit-or-miss generation with a fixed number of
 *    trials. Events have weight `+-1` (overweights `+-|f|/R`) and are not
 *    normalized. The normalization `V / n_trials` is part of the returned
 *    `GenerationResult`. Use `predicted_efficiency()` to choose the number
 *    of trials for a target number of events.
 *
 * Throughout, `f` denotes the integrand multiplied by the weight of the grid
 * mapping.
 *
 * The inheritance is `Derived -> GeneratorBase -> Integrator`, so the
 * generator is also the integrator and the type aliases of the
 * integrator are inherited without ambiguity.
 *
 * The raw envelope table `envelope_` lives in this class: it is allocated
 * with the shape from `env_shape()`, and seeding, scaling, serialization and
 * the checks all happen here, as does storing the volume after sealing. Each
 * entry of the table is one factor of the product `R(u)`, with `ndim` factors
 * per point. How the table is laid out is up to the derived generator, which
 * follows the sampling structure of its integrator (one table per dimension
 * for VEGAS; diagonal and conditional tables in the frozen sampling order for
 * BASIN). `env_propose` has to draw `u` with density `R(u) / V`, map it to
 * the point as `map_point` would, and return `R(u)` (up to rounding), and
 * `env_value(cell)` has to evaluate the same product for the raising passes.
 * The proposal knows the drawn cells, so it maps directly instead of handing
 * `u` to `map_point`, which would have to find the same cells again.
 *
 * Hooks that `Derived` has to provide:
 *  - `make_cell_ctx()`: cell-context buffer for `map_point`;
 *  - `env_shape()`: shape of the table for the current grid;
 *  - `env_value(cell)`: the product R for the cells recorded by `map_point`;
 *  - `env_raise(cell, gamma)`: multiply the cells recorded by `map_point` by `gamma`;
 *  - `env_propose(point)`: draw a point with density `R/V` in u-space using the
 *    integrator RNG, set `point.x` and `point.weight` as `map_point` would, and
 *    return `R(u)`;
 *  - `env_prepare()`: (re)build the caches for the proposal (CDFs) and return `V = int R(u) du`.
 *
 * The table can be replaced in between seals (seeding, loading), so any views
 * or caches of it have to be refreshed in `env_prepare()`.
 *
 * @tparam Derived The derived generator class (e.g., VegasGenerator).
 * @tparam Integrator The integrator the generator extends (e.g., Vegas<>).
 */
template <typename Derived, typename Integrator>
class GeneratorBase : public Integrator {
 public:
  using IntBase = Integrator;
  using typename IntBase::count_type;
  using typename IntBase::int_acc_type;
  using typename IntBase::num_traits;
  using typename IntBase::point_type;
  using typename IntBase::size_type;
  using typename IntBase::value_type;
  using gen_result_type = GenerationResult<value_type, count_type>;
  using env_result_type = EnvelopeResult<value_type, count_type>;

  // shorthands to save typing
  using S = size_type;
  using T = value_type;
  using U = count_type;

  using IntBase::IntBase;

  /// @name Envelope Lifecycle
  /// @{

  /*!
   * @brief Seed a flat envelope with a given value.
   *
   * A natural choice is `|I|` from a frozen production run. Since `|I| <= A`,
   * the envelope starts out low and `optimize_envelope` only raises it where
   * it is violated. If the integrand has large cancellations, `|I|` can be far
   * below `A` and `initialize_envelope(integrand, ncall)` is the better choice.
   *
   * @param abs_integral The seed, usually the absolute integral A or a lower
   *        bound on it. Must be finite and positive.
   * @throws std::invalid_argument if the grid is not frozen or the seed is not
   *         finite and positive.
   */
  void initialize_envelope(T abs_integral) {
    require_frozen("initialize_envelope");
    reset_envelope_state();
    seed_envelope(abs_integral);
  }

  /*!
   * @brief Seed a flat envelope from a new estimate of the absolute integral.
   *
   * Evaluates the integrand at `ncall` points sampled from the frozen grid and
   * seeds the flat envelope with the estimate of A. The estimate is also kept
   * as the A diagnostic (see `abs_integral_estimate()`).
   *
   * @param integrand The integrand.
   * @param ncall The number of samples; must be > 0.
   * @throws std::invalid_argument if the grid is not frozen or ncall == 0.
   * @throws std::runtime_error if the estimated absolute integral is zero (the
   *         integrand vanished or was non-finite at all samples).
   */
  template <typename I>
  void initialize_envelope(I&& integrand, U ncall) {
    require_frozen("initialize_envelope");
    if (ncall == U(0)) {
      throw std::invalid_argument("initialize_envelope requires ncall > 0");
    }
    reset_envelope_state();

    point_type point{derived().ndim(), derived().user_data()};
    std::vector<T> u_buf(derived().ndim());
    auto cell = derived().make_cell_ctx();

    for (U i = 0; i < ncall; ++i) {
      for (S idim = 0; idim < derived().ndim(); ++idim)
        u_buf[idim] = derived().ran();
      point.sample_index = i;
      derived().map_point(u_buf, point, cell);
      T abs_fval = T(0);
      (void)eval_abs(integrand, point, abs_fval, "initialize_envelope");
      accumulate_finite(abs_acc_, abs_fval);
    }
    require_finite_statistics(abs_acc_);
    if (!(abs_acc_.value() > T(0))) {
      throw std::runtime_error(
          "initialize_envelope: sampled abs-integral is zero; cannot seed an envelope");
    }
    seed_envelope(abs_acc_.value());
  }

  /*!
   * @brief A single pass to raise an initialized envelope.
   *
   * Evaluates the integrand at `neval` points sampled from the grid.
   * Whenever `|f|` lies above the envelope, hit cells are raised.
   * Repeated passes until convergence are done by `optimize_envelope`.
   * All samples are also added to the running estimate of A
   * (`abs_integral_estimate()`).
   *
   * @param integrand The integrand.
   * @param neval The number of samples for this pass; must be > 0.
   * @return An `EnvelopeResult` with the running A estimate, the number of
   *         violations in this pass, and the volume of the sealed envelope.
   * @throws std::invalid_argument if the grid is not frozen or neval == 0.
   * @throws std::runtime_error if the envelope was not initialized or does
   *         not match the current grid.
   */
  template <typename I>
  env_result_type raise_envelope(I&& integrand, U neval) {
    require_frozen("raise_envelope");
    if (neval == U(0)) {
      throw std::invalid_argument("raise_envelope requires neval > 0");
    }
    require_ready("raise_envelope");

    const S ndim = derived().ndim();
    point_type point{ndim, derived().user_data()};
    std::vector<T> u_buf(ndim);
    auto cell = derived().make_cell_ctx();

    // raise each factor that was hit such that R grows by at most ~10%
    // @todo: make this more flexible with an option (enum)
    const T gamma = T(1) + T(1) / (T(10) * T(ndim));

    envelope_ready_ = false;  // if the pass throws, we must not be left with stale CDFs
    U n_violations = 0;
    U n_nonfinite = 0;
    for (U i = 0; i < neval; ++i) {
      for (S idim = 0; idim < ndim; ++idim)
        u_buf[idim] = derived().ran();
      point.sample_index = i;
      derived().map_point(u_buf, point, cell);
      T abs_fval = T(0);
      const bool finite = eval_abs(integrand, point, abs_fval, "raise_envelope");
      if (!finite) ++n_nonfinite;
      accumulate_finite(abs_acc_, abs_fval);
      if (finite && abs_fval > derived().env_value(cell)) {
        derived().env_raise(cell, gamma);
        ++n_violations;
      }
    }
    require_finite_statistics(abs_acc_);
    seal_envelope();

    return env_result_type{abs_acc_, n_violations, envelope_volume(), n_nonfinite};
  }

  /*!
   * @brief Raise the envelope until the violation rate is small enough.
   *
   * Runs up to `max_passes` passes of `raise_envelope` with `neval` samples
   * each. Stops early once the violation rate `n_violations / neval` of a pass
   * is at or below `target_rate`.
   *
   * @param integrand The integrand.
   * @param neval The number of samples per pass; must be > 0.
   * @param max_passes The maximum number of passes; must be > 0.
   * @param target_rate Stop once the violation rate of a pass is at or below this.
   * @return The `EnvelopeResult` of the last pass (running A estimate, the
   *         violation count of the last pass, and the sealed volume).
   * @throws std::invalid_argument if the grid is not frozen, neval == 0, or
   *         max_passes == 0.
   * @throws std::runtime_error if the envelope was not initialized or does
   *         not match the current grid.
   */
  template <typename I>
  env_result_type optimize_envelope(I&& integrand, U neval, U max_passes = U(8),
                                    T target_rate = T(1e-3)) {
    if (max_passes == U(0)) {
      throw std::invalid_argument("optimize_envelope requires max_passes > 0");
    }
    env_result_type res{};
    for (U pass = 0; pass < max_passes; ++pass) {
      res = raise_envelope(integrand, neval);
      if (T(res.n_violations()) <= target_rate * T(neval)) break;
    }
    return res;
  }

  /// @}

  /// @name Event Generation
  /// @{

  /*!
   * @brief Generate unnormalized events by hit-or-miss using `ntrials` trials.
   *
   * Points are proposed according to the envelope and accepted events are
   * passed to `event_callback(point, weight)` with weight `+-1`, or `+-|f|/R`
   * for the rare cases where the envelope is violated. The events are not
   * normalized: the signed integral is estimated by
   * `volume() * sum(weights) / n_trials()` and both factors are reported in
   * the returned `GenerationResult`. For a completed run this estimate is
   * unbiased, overweights included.
   *
   * The callback can return an `EventSignal`. Returning `EventSignal::STOP`
   * ends the generation after the current event with a partial result
   * (`status() == STOPPED`), e.g. to stop once enough events were collected.
   * Stopping based on the events can bias the estimates by `O(1/N)`. To aim
   * for `N` events, use `N / predicted_efficiency()` trials instead.
   *
   * By default, trials with a non-finite integrand value are rejected and
   * contribute zero. With `Options::strict_finite_integrand = true` an
   * exception is thrown instead.
   *
   * `point.sample_index` holds the running trial index, which is unique for
   * each integrand call as in the integration, so the callback can use it to
   * track the progress. If `progress_bar` is unset or `true` and verbosity is
   * enabled, a progress bar over the trials is printed to `stderr`.
   *
   * @param integrand The integrand.
   * @param ntrials The number of trials; must be > 0.
   * @param event_callback Called as `event_callback(const point&, weight)` for
   *        each accepted event; may return an `EventSignal` to stop.
   * @return A `GenerationResult` with the statistics of the run.
   * @throws std::invalid_argument if the grid is not frozen, ntrials == 0, or
   *         `progress_step` is invalid while the progress bar is shown.
   * @throws std::runtime_error if the envelope is not ready or does not match
   *         the current grid.
   * @throws std::overflow_error if the weights or their sums overflow.
   */
  template <typename I, typename ECB>
  gen_result_type generate_trials(I&& integrand, U ntrials, ECB&& event_callback) {
    if (ntrials == U(0)) throw std::invalid_argument("generate_trials requires ntrials > 0");
    require_frozen("generate_trials");
    require_ready("generate_trials");

    point_type point{derived().ndim(), derived().user_data()};

    gen_result_type res;
    res.envelope_volume_ = envelope_volume();
    U n_trials = 0;
    // overweights go into `res.acc_` right away; the events of weight +-1 are
    // only counted and added in one go after the loop
    U n_negative_overweight = 0;

    std::optional<util::ProgressBar> bar;
    U milestone_step = 0;
    U next_milestone = ntrials;  // never reached inside the loop without a bar
    const auto update_bar = [&] {
      bar->update(static_cast<double>(n_trials) / static_cast<double>(ntrials),
                  std::format("trials {}/{}", n_trials, ntrials));
    };
    const auto& opts = derived().opts_;
    if (opts.progress_bar.value_or(true) && opts.verbosity.value_or(0) > 0) {
      const double progress_step = opts.progress_step.value_or(DEFAULT_PROGRESS_STEP);
      if (!(progress_step > 0.0) || progress_step > 1.0) {
        throw std::invalid_argument("progress_step must be > 0 and <= 1");
      }
      milestone_step =
          util::math::max(U(1), static_cast<U>(static_cast<double>(ntrials) * progress_step));
      next_milestone = milestone_step;
      bar.emplace();
    }

    while (n_trials < ntrials) {
      // check at the top of the loop so rejected and skipped trials are counted too
      if (n_trials >= next_milestone) [[unlikely]] {
        next_milestone += milestone_step;
        update_bar();
      }
      const T abs_fval_envelope = derived().env_propose(point);
      // check the envelope before calling the integrand; otherwise a
      // non-finite integrand value would `continue` past this check
      if (!(abs_fval_envelope > T(0)) || !std::isfinite(abs_fval_envelope)) {
        throw std::runtime_error("generation: envelope value is not finite and positive");
      }
      point.sample_index = n_trials;
      ++n_trials;
      const T fval = point.weight * integrand(point);
      if (!std::isfinite(fval)) {
        ++res.n_nonfinite_;
        if (strict_finite_integrand()) {
          throw std::runtime_error("generation: non-finite integrand contribution");
        }
        continue;
      }
      const T abs_fval = util::math::abs(fval);
      T event_weight;
      if (abs_fval > abs_fval_envelope) {
        // overweight event: always accept and keep the factor |f|/R in the weight
        const T w_over = abs_fval / abs_fval_envelope;
        res.n_overweight_++;
        res.max_overweight_ = util::math::max(res.max_overweight_, w_over);
        event_weight = T(util::math::sgn(fval)) * w_over;
        accumulate_finite(res.acc_, event_weight);
        if (event_weight < T(0)) n_negative_overweight++;
      } else {
        // rejection sampling: accept with probability |f|/R
        const T r = derived().ran();
        if (r * abs_fval_envelope < abs_fval) {
          event_weight = T(util::math::sgn(fval));
        } else {
          continue;  // rejected: contributes 0, added to the count after the loop
        }
      }
      if (event_weight < T(0)) res.n_negative_++;
      res.n_events_++;
      // pass on the event (read only); the callback can stop the run
      const point_type& event_point = point;
      using cb_result_t = std::invoke_result_t<ECB&, const point_type&, T>;
      if constexpr (std::is_same_v<std::remove_cvref_t<cb_result_t>, EventSignal>) {
        if (has_signal(event_callback(event_point, event_weight), EventSignal::STOP)) {
          res.status_ = GenerationStatus::STOPPED;
          break;
        }
      } else {
        event_callback(event_point, event_weight);
      }
    }  // while
    if (bar) update_bar();

    // add the events of weight +-1: their weights sum to (#positive - #negative)
    // and their squares to their number (exact in T while the counts are below 2^53)
    const U n_unit = res.n_events_ - res.n_overweight_;
    const U n_unit_negative = res.n_negative_ - n_negative_overweight;
    int_acc_type unit_acc;
    unit_acc.reset(T(n_unit - n_unit_negative) - T(n_unit_negative), T(n_unit), n_unit);
    res.acc_.accumulate(unit_acc);
    // rejected trials contribute 0 to the sums; account for them in the count
    res.acc_.accumulate_zeros(n_trials - res.n_events_);
    require_finite_statistics(res.acc_);
    return res;
  }

  /// @}

  /// @name Envelope Queries & Manipulation
  /// @{

  /// @brief Whether the integration grid is frozen (required for generation).
  [[nodiscard]] inline bool is_frozen() const {
    return derived().opts_.frozen.value_or(false);
  }

  /// @brief Whether there is a sealed envelope that matches the current grid,
  ///        i.e. whether `generate_trials` can use it. After the grid was
  ///        adapted this is `false` and the envelope has to be initialized again.
  [[nodiscard]] inline bool envelope_ready() const {
    return envelope_ready_ && envelope_grid_hash_ == derived().hash().value();
  }

  /// @brief The seed the flat envelope was initialized with.
  [[nodiscard]] inline T envelope_seed() const noexcept {
    return envelope_seed_;
  }

  /*!
   * @brief Running estimate of the absolute integral A (diagnostic).
   *
   * Empty (count 0) unless the integrand was sampled with
   * `initialize_envelope(integrand, ncall)` or raising passes.
   */
  [[nodiscard]] inline const int_acc_type& abs_integral_estimate() const noexcept {
    return abs_acc_;
  }

  /// @brief The envelope volume `V = int R(u) du`. Events of a generation run
  ///        are normalized with `V / n_trials`.
  [[nodiscard]] inline T envelope_volume() const noexcept {
    return envelope_volume_;
  }

  /*!
   * @brief Predicted unweighting efficiency `eps = A / V` (accepted events per trial).
   *
   * Uses the estimate of A if the integrand was sampled
   * (`initialize_envelope(integrand, ncall)` or raising passes) and the
   * envelope seed otherwise (a flat envelope that was not scaled then gives
   * exactly 1). The actual efficiency is `int min(|f|, R) du / V <= A / V`,
   * so up to the statistical error on A the prediction is an upper bound and
   * `N / eps` trials will usually give slightly fewer than `N` events. The
   * ratio is not clamped: a value above 1 means that the envelope lies below
   * the integrand on average.
   *
   * @throws std::runtime_error if the envelope is not ready or does not match
   *         the current grid, or if the estimate of A is not finite and positive.
   */
  [[nodiscard]] T predicted_efficiency() const {
    require_ready("predicted_efficiency");
    const T a_est = abs_acc_.count() > U(0) ? abs_acc_.value() : envelope_seed_;
    if (!(a_est > T(0)) || !std::isfinite(a_est)) {
      throw std::runtime_error(
          "predicted_efficiency requires a finite positive abs-integral estimate");
    }
    return a_est / envelope_volume_;
  }

  /*!
   * @brief Multiply the whole envelope by `factor` (e.g. a safety factor).
   *
   * @param factor The scaling factor; must be finite and positive.
   * @throws std::invalid_argument if the factor is invalid or the grid is not frozen.
   * @throws std::runtime_error if the envelope was not initialized or does
   *         not match the current grid.
   */
  inline void envelope_scale(T factor) {
    require_frozen("envelope_scale");
    if (!(factor > T(0)) || !std::isfinite(factor)) {
      throw std::invalid_argument("envelope_scale requires a finite positive factor");
    }
    // sealing binds the envelope to the current grid hash, so we must not do
    // that for an envelope that was raised on an older grid
    require_ready("envelope_scale");
    envelope_ready_ = false;
    // spread evenly across the ndim factors of R
    const T per = std::pow(factor, T(1) / T(derived().ndim()));
    for (auto& v : envelope_)
      v *= per;
    seal_envelope();
  }

  /// @}

  /// @name State Persistence
  /// @{

  /*!
   * @brief Save the generator state (integrator state + envelope) to a file.
   *
   * Writes the same `.khs` format as `save` of the integrator and appends the
   * envelope as an extra block at the end. The plain integrator can therefore
   * still read the grid from a generator file.
   *
   * @param filepath The path to the file where the state should be saved.
   */
  void save(const std::filesystem::path& filepath) const {
    this->write_file(filepath, detail::FileType::STATE,
                     [this](std::ostream& out) { write_state_stream(out); });
  }

  /*!
   * @brief Save the generator state to the default state file.
   * @return The path to the saved state file.
   */
  std::filesystem::path save() const {
    std::filesystem::path fstate = this->file_state();
    save(fstate);
    return fstate;
  }

  /*!
   * @brief Load the generator state (integrator state + envelope) from a file.
   *
   * A file without an envelope block at the end (e.g. written by the plain
   * integrator) only loads the integrator state and leaves the envelope
   * uninitialized.
   *
   * @param filepath The path to the file from which the state should be loaded.
   */
  void load(const std::filesystem::path& filepath) {
    if (!this->state_file_exists(filepath)) return;
    this->read_file(filepath, detail::FileType::STATE,
                    [this](std::istream& in) { read_state_stream(in); });
  }

  /*!
   * @brief Load the generator state from the default state file.
   * @return The path to the loaded state file.
   */
  std::filesystem::path load() {
    std::filesystem::path fstate = this->file_state();
    load(fstate);
    return fstate;
  }

  /// @name State Streams (integrator state + envelope block)
  /// @{

  /// @brief Writes the integrator state followed by the envelope block.
  void write_state_stream(std::ostream& out) const {
    IntBase::write_state_stream(out);
    write_envelope_stream(out);
  }

  /// @brief Reads the integrator state and, if present, the envelope block.
  void read_state_stream(std::istream& in) {
    // reset first so a failed read does not leave a ready envelope behind
    reset_envelope_state();
    IntBase::read_state_stream(in);
    read_envelope_stream(in);
  }

  /// @}

 protected:
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
  void require_frozen(std::string_view what) const {
    if (!is_frozen()) {
      throw std::invalid_argument(std::format("{} requires a frozen integration grid", what));
    }
  }

  /// @brief Require a sealed envelope that matches the current grid.
  void require_ready(std::string_view what) const {
    if (!envelope_ready_) {
      throw std::runtime_error(
          std::format("{} requires an envelope seeded by initialize_envelope", what));
    }
    if (envelope_grid_hash_ != derived().hash().value()) {
      throw std::runtime_error(
          std::format("{}: envelope does not match the current grid; re-initialize", what));
    }
  }

  /// @brief Evaluate |f| at `point` and check that it is finite.
  template <typename I>
  bool eval_abs(I& integrand, point_type& point, T& abs_fval, std::string_view what) const {
    const T fval = point.weight * integrand(point);
    if (!std::isfinite(fval)) {
      if (strict_finite_integrand()) {
        throw std::runtime_error(std::format("{}: non-finite integrand contribution", what));
      }
      abs_fval = T(0);
      return false;
    }
    abs_fval = util::math::abs(fval);
    return true;
  }

  [[nodiscard]] bool strict_finite_integrand() const noexcept {
    return derived().opts_.strict_finite_integrand.value_or(false);
  }

  /// @brief Accumulate `value` after checking that it and its square are finite.
  ///        An overflow of the running sums is checked only once at the end of
  ///        the run in `require_finite_statistics`.
  static void accumulate_finite(int_acc_type& acc, T value) {
    const T square = value * value;
    if (!std::isfinite(square)) {  // also catches a non-finite value
      throw std::overflow_error(
          "generation: contribution or square overflows; rebuild the envelope "
          "from an absolute-integral estimate or rescale the integrand");
    }
    acc.accumulate(value, square);
  }

  /// @brief Throw if the accumulated sums overflowed. If the sums are finite,
  ///        so are the value and the variance.
  static void require_finite_statistics(const int_acc_type& acc) {
    if (!std::isfinite(acc.f_.result()) || !std::isfinite(acc.f2_.result())) {
      throw std::overflow_error("generation: accumulated contributions overflow");
    }
  }

  /// @brief Clear all envelope state before (re-)seeding.
  void reset_envelope_state() {
    envelope_ready_ = false;
    envelope_grid_hash_ = {};
    envelope_seed_ = T(0);
    envelope_volume_ = T(0);
    abs_acc_.reset();
  }

  /// @brief Seed a flat envelope `R(u) = seed` and seal it.
  void seed_envelope(T seed) {
    if (!(seed > T(0)) || !std::isfinite(seed)) {
      throw std::invalid_argument("initialize_envelope requires a finite seed > 0");
    }
    // spread evenly across the ndim factors of R
    // (the root of a finite positive seed stays finite and positive)
    const T per = std::pow(seed, T(1) / T(derived().ndim()));
    envelope_ = ndarray::NDArray<T, S>(derived().env_shape());
    envelope_.fill(per);
    envelope_seed_ = seed;
    seal_envelope();
  }

  /// @brief Prepare the envelope for generation (CDFs, normalization), check
  ///        its volume, and bind it to `grid_hash`.
  inline void seal_envelope(kakuhen::util::HashValue_t grid_hash) {
    envelope_ready_ = false;
    envelope_volume_ = T(0);
    const T vol = derived().env_prepare();
    if (!(vol > T(0)) || !std::isfinite(vol)) {
      throw std::runtime_error("generation envelope has invalid volume");
    }
    envelope_volume_ = vol;
    envelope_grid_hash_ = grid_hash;
    envelope_ready_ = true;
  }

  /// @brief Seal the envelope and bind it to the current grid.
  inline void seal_envelope() {
    seal_envelope(derived().hash().value());
  }

  /*!
   * @brief Append the envelope block to a state stream.
   *
   * The block starts with a tag and a version, followed by the ready flag, the
   * seed, the grid hash, the A accumulator, and the raw envelope table.
   * Anything derived from the table (CDFs, subtree integrals, volume) is not
   * written and gets rebuilt when the envelope is sealed after loading.
   */
  void write_envelope_stream(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    write_bytes(out, envelope_block_tag.data(), envelope_block_tag.size());
    serialize_one<uint8_t>(out, envelope_block_version);
    serialize_one<uint8_t>(out, envelope_ready_ ? uint8_t(1) : uint8_t(0));
    serialize_one<T>(out, envelope_seed_);
    serialize_one<kakuhen::util::HashValue_t>(out, envelope_grid_hash_);
    abs_acc_.serialize(out);
    write_envelope_table(out);
  }

  /*!
   * @brief Read the envelope block from a state stream if there is one.
   *
   * Assumes that `read_state_stream` already reset the envelope state. If the
   * stream ends right after the integrator state (plain integrator file), the
   * envelope stays uninitialized. Otherwise the tag and version are checked,
   * the raw table is read, and a ready envelope is sealed again with the
   * stored grid hash. That way, an envelope that was stale when it was saved
   * is still stale after loading.
   *
   * @throws std::runtime_error if the envelope block is corrupt or has an
   *         unsupported version.
   */
  void read_envelope_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    if (in.peek() == std::istream::traits_type::eof()) {
      // plain integrator file: no envelope block
      envelope_ = {};
      derived().print_info_message("state",
                                   "no envelope block in state stream; "
                                   "envelope left uninitialized");
      return;
    }
    std::array<char, envelope_block_tag.size()> tag{};
    read_bytes(in, tag.data(), tag.size());
    if (std::string_view(tag.data(), tag.size()) != envelope_block_tag) {
      throw std::runtime_error("invalid envelope block tag in state stream");
    }
    uint8_t version;
    deserialize_one<uint8_t>(in, version);
    if (version != envelope_block_version) {
      throw std::runtime_error("unsupported envelope block version");
    }
    uint8_t ready;
    deserialize_one<uint8_t>(in, ready);
    if (ready > uint8_t(1)) {
      throw std::runtime_error("corrupt envelope block: invalid ready flag");
    }
    deserialize_one<T>(in, envelope_seed_);
    kakuhen::util::HashValue_t grid_hash;
    deserialize_one<kakuhen::util::HashValue_t>(in, grid_hash);
    abs_acc_.deserialize(in);
    read_envelope_table(in);
    if (ready != 0) seal_envelope(grid_hash);
  }

 protected:
  /// @brief Write the expected table shape followed by the raw envelope table.
  ///        If the table was never seeded for this grid, zeros are written instead.
  void write_envelope_table(std::ostream& out) const {
    using namespace kakuhen::util::serialize;
    const auto shape = derived().env_shape();
    for (const S extent : shape)
      serialize_one<S>(out, extent);
    if (std::ranges::equal(envelope_.shape(), shape)) {
      envelope_.serialize(out);
    } else {
      ndarray::NDArray<T, S> zeros(shape);
      zeros.fill(T(0));
      zeros.serialize(out);
    }
  }

  /// @brief Read the raw envelope table. Its shape has to match the integrator
  ///        state that was already loaded. The proposal caches are rebuilt when sealing.
  void read_envelope_table(std::istream& in) {
    using namespace kakuhen::util::serialize;
    const auto shape = derived().env_shape();
    for (const S extent : shape) {
      S extent_chk;
      deserialize_one<S>(in, extent_chk);
      if (extent_chk != extent) {
        throw std::runtime_error("envelope table dimensions do not match the integrator state");
      }
    }
    envelope_.deserialize(in);
    // deserialize takes the shape from the stream, so a corrupt block could
    // pass the dimension check above and still have a different shape
    if (!std::ranges::equal(envelope_.shape(), shape)) {
      throw std::runtime_error("corrupt envelope block: unexpected table shape");
    }
  }

  /// the raw envelope table; the `ndim` factors for a point multiply to `R(u)`
  ndarray::NDArray<T, S> envelope_;

 private:
  /// tag and version of the envelope block at the end of `.khs` state streams
  static constexpr std::string_view envelope_block_tag = "KHENVB";  // 6 bytes
  static constexpr uint8_t envelope_block_version = 1;

  /// running estimate of the absolute integral A (diagnostic)
  int_acc_type abs_acc_;
  /// the seed the flat envelope was initialized with
  T envelope_seed_{};
  /// volume of the sealed envelope `V = int R(u) du`
  T envelope_volume_{};
  bool envelope_ready_ = false;
  kakuhen::util::HashValue_t envelope_grid_hash_{};
};

}  // namespace kakuhen::integrator
