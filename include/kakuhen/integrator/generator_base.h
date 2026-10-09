#pragma once

#include "kakuhen/integrator/detail/envelope_data.h"
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
#include "kakuhen/util/user_data.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <istream>
#include <limits>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace kakuhen::integrator {

/// @brief how a generation run ended; a merged result is STOPPED if any run stopped early
enum class GenerationStatus : uint8_t {
  COMPLETED = 0,  //!< all trials used up
  STOPPED = 1,    //!< the event callback returned `EventSignal::STOP`
};

/// @brief status name, e.g. "completed"
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
 * @brief statistics of a generation run
 *
 * Events are not normalized: they carry weight +-1, or +-|f|/R for
 * overweight events. The normalization is `volume() / n_trials()` and both
 * factors are stored here. The accumulator collects the signed weight of
 * every trial (0 for rejected ones), so `value() = volume() * <w>` estimates
 * the signed integral. Unbiased for a completed run.
 *
 * Results of independent runs with the same envelope can be merged with
 * `accumulate()`.
 *
 * @tparam T value type of the integral estimates (e.g. double)
 * @tparam U count type for trials and events (e.g. uint64_t)
 */
template <typename T, typename U>
struct GenerationResult {
  using value_type = T;
  using count_type = U;
  using int_acc_type = IntegralAccumulator<T, U>;

  int_acc_type acc_{};        //!< signed weight of each trial (0 if rejected)
  U n_events_ = 0;            //!< accepted events, overweights included
  U n_overweight_ = 0;        //!< accepted events above the envelope
  U n_negative_ = 0;          //!< accepted events with negative weight
  U n_nonfinite_ = 0;         //!< trials with a non-finite integrand (set to zero)
  T max_overweight_ = T(0);   //!< largest overweight factor |f|/R (0 if none)
  T envelope_volume_ = T(0);  //!< envelope volume V used for the run
  GenerationStatus status_ = GenerationStatus::COMPLETED;  //!< how the run ended

  /// @name Queries
  /// @{

  /// @brief total number of trials, accepted or not
  [[nodiscard]] inline U n_trials() const noexcept {
    return acc_.count();
  }
  /// @brief number of accepted events (overweights included)
  [[nodiscard]] inline U n_events() const noexcept {
    return n_events_;
  }
  /// @brief number of rejected trials
  [[nodiscard]] inline U n_rejected() const noexcept {
    return n_trials() - n_events_;
  }
  /// @brief number of accepted events above the envelope (|weight| > 1)
  [[nodiscard]] inline U n_overweight() const noexcept {
    return n_overweight_;
  }
  /// @brief number of accepted events with negative weight
  [[nodiscard]] inline U n_negative() const noexcept {
    return n_negative_;
  }
  /// @brief number of trials with a non-finite integrand (set to zero)
  [[nodiscard]] inline U n_nonfinite() const noexcept {
    return n_nonfinite_;
  }
  /// @brief largest overweight factor |f|/R (0 if none)
  [[nodiscard]] inline T max_overweight() const noexcept {
    return max_overweight_;
  }
  /// @brief envelope volume V; events are normalized with `volume() / n_trials()`
  [[nodiscard]] inline T volume() const noexcept {
    return envelope_volume_;
  }
  /// @brief how the run ended; a merged result is only COMPLETED if all runs were
  [[nodiscard]] inline GenerationStatus status() const noexcept {
    return status_;
  }

  /// @brief unweighting efficiency (accepted events per trial)
  [[nodiscard]] inline T efficiency() const noexcept {
    return n_trials() > U(0) ? T(n_events_) / T(n_trials()) : T(0);
  }
  /// @brief fraction of accepted events with negative weight
  [[nodiscard]] inline T negative_fraction() const noexcept {
    return n_events_ > U(0) ? T(n_negative_) / T(n_events_) : T(0);
  }

  /*!
   * @brief estimate of the signed integral
   *
   * Computed as `V * <w>` over all trials. Unbiased for a completed run,
   * overweights included.
   *
   * @throws std::runtime_error if no trials were accumulated
   */
  [[nodiscard]] T value() const {
    if (n_trials() == U(0)) throw std::runtime_error("no trials accumulated");
    return envelope_volume_ * acc_.value();
  }
  /*!
   * @brief error (standard deviation) of `value()`
   * @throws std::runtime_error if no trials were accumulated
   */
  [[nodiscard]] T error() const {
    if (n_trials() == U(0)) throw std::runtime_error("no trials accumulated");
    return envelope_volume_ * acc_.error();
  }

  /*!
   * @brief normalization per event `volume() / n_trials()`
   *
   * Multiply the event weights by this to get a normalized sample (e.g. to
   * fill histograms). When combining events of several runs, use the
   * normalization of the merged result. Stopping based on accepted events
   * can bias it; use a fixed trial budget if the absolute normalization
   * matters.
   *
   * @throws std::runtime_error if no trials were accumulated
   */
  [[nodiscard]] T normalization() const {
    if (n_trials() == U(0)) throw std::runtime_error("no trials accumulated");
    return envelope_volume_ / T(n_trials());
  }

  /// @}

  /*!
   * @brief merge the statistics of another run that used the same envelope
   *
   * Leaves the result unchanged on failure.
   *
   * @param other the result to merge in
   * @throws std::invalid_argument if both results are non-empty but have
   *         different envelope volumes
   * @throws std::overflow_error if the merged sums overflow
   */
  void accumulate(const GenerationResult<T, U>& other) {
    // validate volume and sums before touching anything
    if (n_trials() > U(0) && other.n_trials() > U(0) &&
        envelope_volume_ != other.envelope_volume_) {
      throw std::invalid_argument(
          "cannot merge runs generated against different envelopes");
    }
    int_acc_type merged_acc = acc_;
    merged_acc.accumulate(other.acc_);
    if (!merged_acc.is_finite()) {
      throw std::overflow_error("merged contributions overflow");
    }
    // an empty run may still have stopped early
    status_ = static_cast<GenerationStatus>(
        util::math::max(static_cast<uint8_t>(status_), static_cast<uint8_t>(other.status_)));
    if (other.n_trials() == U(0)) return;
    if (n_trials() == U(0)) envelope_volume_ = other.envelope_volume_;
    acc_ = merged_acc;
    n_events_ += other.n_events_;
    n_overweight_ += other.n_overweight_;
    n_negative_ += other.n_negative_;
    n_nonfinite_ += other.n_nonfinite_;
    max_overweight_ = util::math::max(max_overweight_, other.max_overweight_);
  }

};  // struct GenerationResult

/// @brief why envelope training or collection returned
enum class EnvelopeStatus : uint8_t {
  BUDGET_EXHAUSTED,     //!< all samples used
  TARGET_REACHED,       //!< `optimize_envelope` reached its target violation rate
  RECORD_LIMIT_REACHED  //!< `collect_envelope` hit its record limit
};

/// @brief status name, e.g. "budget_exhausted"
constexpr std::string_view to_string(EnvelopeStatus status) noexcept {
  switch (status) {
    case EnvelopeStatus::BUDGET_EXHAUSTED:
      return "budget_exhausted";
    case EnvelopeStatus::TARGET_REACHED:
      return "target_reached";
    case EnvelopeStatus::RECORD_LIMIT_REACHED:
      return "record_limit_reached";
  }
  return "unknown";
}

/*!
 * @brief statistics of envelope training, collection, adaptation or merging
 *
 * The estimate of the absolute integral A = int |f| du includes all
 * accumulated samples. The counts refer to this call, or to the batch
 * processed by `adapt_envelope` or `merge_envelope`. A violation is an
 * evaluation whose |f| exceeded the envelope as it stood when checked, so
 * `n_violations() / n_evaluations()` is the violation rate of every call.
 * `n_raised()` counts the envelope updates: equal to `n_violations()` for
 * raising passes, zero for collection and possibly smaller on replay.
 *
 * @tparam T value type of the integral estimates (e.g. double)
 * @tparam U count type for evaluations (e.g. uint64_t)
 */
template <typename T, typename U>
struct EnvelopeResult {
  using value_type = T;
  using count_type = U;
  using int_acc_type = IntegralAccumulator<T, U>;

  int_acc_type abs_acc_{};    //!< running estimate of the absolute integral A
  U n_violations_ = 0;        //!< evaluations above the envelope when checked
  U n_raised_ = 0;            //!< envelope raises in this call
  T envelope_volume_ = T(0);  //!< envelope volume V after sealing
  U n_nonfinite_ = 0;         //!< evaluations with a non-finite integrand (set to zero)
  U n_evaluations_ = 0;       //!< evaluations in this call or the processed batch
  EnvelopeStatus status_ = EnvelopeStatus::BUDGET_EXHAUSTED;  //!< why the call returned

  /// @name Queries
  /// @{

  /// @brief estimate of the absolute integral A = int |f| du
  [[nodiscard]] inline T abs_integral() const noexcept {
    return abs_acc_.value();
  }
  /// @brief error (standard deviation) of `abs_integral()`
  [[nodiscard]] inline T abs_error() const noexcept {
    return abs_acc_.error();
  }
  /// @brief number of samples in the A estimate
  [[nodiscard]] inline U count() const noexcept {
    return abs_acc_.count();
  }
  /// @brief evaluations whose |f| exceeded the envelope as it stood when checked
  [[nodiscard]] inline U n_violations() const noexcept {
    return n_violations_;
  }
  /// @brief envelope raises, at most one per violation
  [[nodiscard]] inline U n_raised() const noexcept {
    return n_raised_;
  }
  /// @brief evaluations with a non-finite integrand (set to zero);
  ///        always zero for `adapt_envelope` and `merge_envelope` as they don't sample
  [[nodiscard]] inline U n_nonfinite() const noexcept {
    return n_nonfinite_;
  }
  /// @brief evaluations in this call, or in the batch being adapted or merged
  [[nodiscard]] inline U n_evaluations() const noexcept {
    return n_evaluations_;
  }
  /// @brief why the call returned
  [[nodiscard]] inline EnvelopeStatus status() const noexcept {
    return status_;
  }
  /// @brief volume V of the sealed envelope
  [[nodiscard]] inline T volume() const noexcept {
    return envelope_volume_;
  }
  /// @brief predicted unweighting efficiency A / V
  [[nodiscard]] inline T efficiency() const noexcept {
    return abs_integral() / envelope_volume_;
  }

  /// @}

};  // struct EnvelopeResult

namespace detail {

/// @brief add `weight` to an inclusive CDF sum
/// @throws std::runtime_error unless the result is finite and strictly increasing
template <typename T>
T cdf_step(T sum, T weight) {
  const T next = sum + weight;
  if (!(next > sum) || !std::isfinite(next)) [[unlikely]]
    throw std::runtime_error("envelope weights must form a finite strictly increasing CDF");
  return next;
}

/// @brief a cell drawn from an inclusive CDF
template <typename T, typename S>
struct CDFDraw {
  S cell;      //!< the selected cell
  T fraction;  //!< position within the CDF segment of the cell, in [0, 1]
  T width;     //!< CDF mass of the cell `cdf[cell] - cdf[cell - 1]`
};

/// @brief invert an inclusive CDF at `u` in [0, 1]
/// @pre `cdf` non-empty, positive and strictly increasing (as built by `cdf_step`)
template <typename T, typename S>
CDFDraw<T, S> sample_cdf(std::span<const T> cdf, T u) {
  assert(!cdf.empty());
  T target = u * cdf.back();
  // stay strictly below the total, else upper_bound returns end() (cell out of range)
  if (target >= cdf.back()) [[unlikely]]
    target = std::nextafter(cdf.back(), T(0));
  const auto cell = static_cast<S>(std::upper_bound(cdf.begin(), cdf.end(), target) - cdf.begin());
  const T low = cell == 0 ? T(0) : cdf[cell - 1];
  const T width = cdf[cell] - low;
  return {cell, (target - low) / width, width};
}

}  // namespace detail

/*!
 * @brief CRTP base for event generation on a frozen grid
 *
 * The usual sequence is:
 *
 * 1. `initialize_envelope(...)`: seed a flat envelope, either with a given
 *    value or by estimating the absolute integral
 * 2. `optimize_envelope(...)`: raise the envelope until the target violation
 *    rate is reached (or `raise_envelope(...)` for a single pass)
 * 3. `generate_trials(...)`: generate events with a fixed trial budget; the
 *    returned `GenerationResult` supplies the normalization V / n_trials
 *
 * For distributed training, `collect_envelope` records violations without
 * changing the envelope. Apply them locally with `adapt_envelope`, or save
 * them with `save_envelope` and combine the files with `merge_envelope`.
 *
 * Throughout, f is the integrand times the weight of the grid mapping.
 * A non-finite f counts as zero, or throws with
 * `Options::strict_finite_integrand`.
 *
 * This class owns the envelope factors, statistics and pending observations.
 * The derived generator defines the table layout and proposal sampling.
 * At each point, R(u) is the product of `ndim` factors picked from the table
 * by `env_indices`.
 *
 * Required hooks in `Derived`:
 *  - `make_cell_ctx()`: cell-context buffer for `map_point`
 *  - `env_shape()`: shape of the table for the current grid
 *  - `env_indices(cell, indices)`: flat table index of each of the `ndim`
 *    factors of R for the cells recorded by `map_point`
 *  - `env_propose(point)`: draw a point with density R/V in u-space using the
 *    integrator RNG, set `point.x` and `point.weight` as `map_point` would,
 *    and return R(u)
 *  - `env_prepare(table) const`: build the proposal caches (CDFs) for `table`
 *    and return the pair `{V = int R(u) du, caches}`
 *  - `env_commit_cache(caches) noexcept`: install caches built by `env_prepare`
 *
 * `env_grid_hash()` identifies the sampling map the envelope was trained on.
 * Defaults to the grid hash; BASIN also folds in its sampling order.
 *
 * @tparam Derived the derived generator class (e.g. VegasGenerator)
 * @tparam Integrator the integrator the generator extends (e.g. Vegas<>)
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
   * @brief seed a flat envelope with a given value
   *
   * |I| from a frozen production run works as a seed. For integrands with
   * large cancellations, better use `initialize_envelope(integrand, ncall)`
   * to estimate the absolute integral directly.
   *
   * @param abs_integral the seed, usually A or a lower bound on it; must be
   *        finite and positive
   * @throws std::invalid_argument if the grid is not frozen or the seed is
   *         not finite and positive
   * @throws std::runtime_error if envelope data are pending
   */
  void initialize_envelope(T abs_integral) {
    require_frozen();
    require_no_envelope_data();
    reset_envelope_state();
    seed_envelope(abs_integral);
  }

  /*!
   * @brief seed a flat envelope from a fresh estimate of the absolute integral
   *
   * Evaluates the integrand at `ncall` points sampled from the frozen grid
   * and uses the A estimate as the seed and initial `abs_integral_estimate()`.
   *
   * @param integrand the integrand
   * @param ncall number of samples, must be > 0
   * @throws std::invalid_argument if the grid is not frozen or ncall == 0
   * @throws std::runtime_error if envelope data are pending, the estimated
   *         absolute integral is zero, or f is non-finite in strict mode
   * @throws std::overflow_error if the accumulated sums overflow
   */
  template <typename I>
  void initialize_envelope(I&& integrand, U ncall) {
    require_frozen();
    require_no_envelope_data();
    if (ncall == U(0)) {
      throw std::invalid_argument("initialize_envelope requires ncall > 0");
    }
    reset_envelope_state();
    for_each_abs_sample(integrand, ncall, [&](T abs_fval, bool, auto) {
      accumulate_checked(abs_acc_, abs_fval);
      return true;
    });
    require_finite_sums(abs_acc_);
    if (!(abs_acc_.value() > T(0))) {
      throw std::runtime_error(
          "sampled abs-integral is zero; cannot seed an envelope");
    }
    seed_envelope(abs_acc_.value());
  }

  /*!
   * @brief single pass to raise an initialized envelope
   *
   * Evaluates the integrand at `neval` points sampled from the grid and
   * raises the hit cells whenever |f| lies above the envelope.
   * `optimize_envelope` repeats this until convergence. All samples also go
   * into the running A estimate (`abs_integral_estimate()`).
   *
   * The pass trains a copy of the envelope, so on failure the envelope, its
   * proposal caches and the A estimate are unchanged (the RNG has advanced
   * though).
   *
   * @param integrand the integrand
   * @param neval number of samples for this pass, must be > 0
   * @return running A estimate, violations in this pass and volume of the
   *         sealed envelope
   * @throws std::invalid_argument if the grid is not frozen or neval == 0
   * @throws std::runtime_error if the envelope is not initialized or does not
   *         match the current grid, or f is non-finite in strict mode
   * @throws std::overflow_error if the evaluation count or sums overflow
   */
  template <typename I>
  env_result_type raise_envelope(I&& integrand, U neval) {
    require_frozen();
    if (neval == U(0)) {
      throw std::invalid_argument("raise_envelope requires neval > 0");
    }
    require_envelope_ready();
    require_count_capacity(neval);

    // each violation raises R by (1 + 1 / (10 * ndim))^ndim, about 10%
    const T gamma = envelope_raising_factor();
    env_result_type result;
    // we create local copies to enable clean roll-backs in case of errors
    ndarray::NDArray<T, S> candidate(envelope_.shape());
    std::ranges::copy(envelope_, candidate.begin());
    auto next_abs = abs_acc_;
    for_each_abs_sample(integrand, neval, [&](T abs_fval, bool finite, auto indices) {
      accumulate_checked(next_abs, abs_fval);
      count_evaluation(result, finite);
      // a non-finite sample has `abs_fval = 0` and never raises
      if (detail::raise_envelope_record(candidate, indices, abs_fval, gamma))
        ++result.n_violations_;
      return true;
    });
    result.n_raised_ = result.n_violations_;  // every violation raises once
    require_finite_sums(next_abs);
    seal_envelope(candidate, derived().env_grid_hash());
    abs_acc_ = next_abs;
    result.abs_acc_ = abs_acc_;
    result.envelope_volume_ = envelope_volume();
    return result;
  }

  /*!
   * @brief record violations without changing the envelope
   *
   * Evaluates the integrand at up to `neval` points sampled from the grid and
   * appends every |f| above the fixed envelope, with its factor indices, to
   * the pending batch. All samples go into the batch statistics and
   * `abs_integral_estimate()`. Apply the batch locally with `adapt_envelope()`,
   * or save it with `save_envelope()` to be merged elsewhere. If an
   * evaluation throws or its statistics overflow, the batch keeps the samples
   * before it.
   *
   * @param integrand the integrand
   * @param neval maximum number of samples, must be > 0
   * @param max_records stop once the pending batch holds this many records
   *        (counting those of earlier calls)
   * @return counts for this call; status RECORD_LIMIT_REACHED if collection
   *         stopped at `max_records`
   * @throws std::invalid_argument if the grid is not frozen or neval == 0
   * @throws std::runtime_error if the envelope is not initialized or does not
   *         match the current grid, or f is non-finite in strict mode
   * @throws std::overflow_error if the evaluation count or sums overflow
   */
  template <typename I>
  env_result_type collect_envelope(I&& integrand, U neval,
                                   U max_records = std::numeric_limits<U>::max()) {
    require_frozen();
    if (neval == U(0)) {
      throw std::invalid_argument("collect_envelope requires neval > 0");
    }
    require_envelope_ready();
    require_count_capacity(neval);

    const auto below_limit = [&] { return n_envelope_records() < max_records; };
    env_result_type result;
    if (below_limit()) {
      for_each_abs_sample(integrand, neval, [&](T abs_fval, bool finite, auto indices) {
        // check before appending so a failed sample leaves the batch intact
        const T square = checked_square(abs_fval);
        auto next_abs = abs_acc_;
        auto next_batch = envelope_data_.acc;
        next_abs.accumulate(abs_fval, square);
        next_batch.accumulate(abs_fval, square);
        require_finite_sums(next_abs);
        require_finite_sums(next_batch);
        if (abs_fval > detail::envelope_bound(envelope_, indices)) {
          envelope_data_.append(abs_fval, indices);
          ++result.n_violations_;
        }
        abs_acc_ = next_abs;  // nothing from here on throws
        envelope_data_.acc = next_batch;
        count_evaluation(result, finite);
        return below_limit();
      });
    }
    if (!below_limit()) result.status_ = EnvelopeStatus::RECORD_LIMIT_REACHED;
    result.abs_acc_ = abs_acc_;
    result.envelope_volume_ = envelope_volume();
    return result;
  }

  /*!
   * @brief raise the envelope until the violation rate is small enough
   *
   * Runs up to `max_passes` passes of `raise_envelope` with `neval` samples
   * each. Stops early once the violation rate `n_violations / neval` of a
   * pass is at or below `target_rate`.
   *
   * @param integrand the integrand
   * @param neval number of samples per pass, must be > 0
   * @param max_passes maximum number of passes, must be > 0
   * @param target_rate stop once a pass is at or below this rate; must be in
   *        [0, 1]. With 0, passes continue until one has no violation; with
   *        1, exactly one pass is run.
   * @return running A estimate, final volume, counts summed over all passes
   *         and whether the target was reached
   * @throws std::invalid_argument if the grid is not frozen, neval == 0,
   *         max_passes == 0, or target_rate is not in [0, 1] (or NaN)
   * @throws std::runtime_error if the envelope is not initialized or does not
   *         match the current grid, or f is non-finite in strict mode
   * @throws std::overflow_error if the evaluation count or sums overflow
   */
  template <typename I>
  env_result_type optimize_envelope(I&& integrand, U neval, U max_passes = U(8),
                                    T target_rate = T(1e-3)) {
    if (max_passes == U(0)) {
      throw std::invalid_argument("optimize_envelope requires max_passes > 0");
    }
    // negated so that NaN is rejected as well
    if (!(target_rate >= T(0) && target_rate <= T(1))) {
      throw std::invalid_argument("optimize_envelope requires a target_rate in [0, 1]");
    }
    env_result_type res;
    for (U pass = 0; pass < max_passes; ++pass) {
      const auto step = raise_envelope(integrand, neval);
      res.abs_acc_ = step.abs_acc_;
      res.envelope_volume_ = step.envelope_volume_;
      res.n_evaluations_ += step.n_evaluations_;
      res.n_violations_ += step.n_violations_;
      res.n_raised_ += step.n_raised_;
      res.n_nonfinite_ += step.n_nonfinite_;
      if (T(step.n_violations_) <= target_rate * T(neval)) {
        res.status_ = EnvelopeStatus::TARGET_REACHED;
        break;
      }
    }
    return res;
  }

  /// @brief whether a collected batch is pending (even one without violations)
  [[nodiscard]] bool has_envelope_data() const noexcept {
    return envelope_data_.present();
  }

  /// @brief number of pending violations across all collection calls
  [[nodiscard]] U n_envelope_records() const noexcept {
    return static_cast<U>(envelope_data_.values.size());
  }

  /// @brief drop pending observations and batch statistics; keeps envelope and A estimate
  void clear_envelope_data() noexcept {
    envelope_data_.clear();
  }

  /*!
   * @brief raise the envelope with the pending observations, then clear the batch
   *
   * Processes observations in collection order, at most one raise each.
   * Doesn't evaluate the integrand and leaves the A estimate alone. On
   * failure, the envelope and pending batch are unchanged.
   *
   * @return the batch's evaluations and records (`n_violations`), number of
   *         raises, current A estimate and envelope volume
   * @throws std::invalid_argument if the grid is not frozen
   * @throws std::runtime_error if the envelope is not initialized or does not
   *         match the current grid
   */
  env_result_type adapt_envelope() {
    require_frozen();
    require_envelope_ready();
    env_result_type result{.abs_acc_ = abs_acc_,
                           .n_violations_ = n_envelope_records(),
                           .envelope_volume_ = envelope_volume(),
                           .n_evaluations_ = envelope_data_.acc.count()};
    if (!has_envelope_data()) return result;
    ndarray::NDArray<T, S> candidate(envelope_.shape());
    std::ranges::copy(envelope_, candidate.begin());
    result.n_raised_ = apply_envelope_records(candidate, envelope_data_);
    seal_envelope(candidate, derived().env_grid_hash());
    clear_envelope_data();
    result.envelope_volume_ = envelope_volume();
    return result;
  }

  /// @}

  /// @name Event Generation
  /// @{

  /*!
   * @brief generate unnormalized events by hit-or-miss using `ntrials` trials
   *
   * Points are proposed according to the envelope and accepted events are
   * passed to `event_callback(point, weight)` with weight +-1, or +-|f|/R
   * where the envelope is violated. The signed integral is estimated by
   * `volume() * sum(weights) / n_trials()`, with both factors reported in the
   * returned `GenerationResult`. Unbiased for a completed run, overweights
   * included.
   *
   * The callback can return an `EventSignal`. `EventSignal::STOP` ends the
   * run after the current event with a partial result (STOPPED), e.g. once
   * enough events were collected. Stopping based on the events can bias the
   * estimates though; to aim for N events, rather use
   * `N / predicted_efficiency()` trials.
   *
   * Trials with a non-finite integrand are rejected and contribute zero;
   * with `Options::strict_finite_integrand` they throw instead.
   *
   * `point.sample_index` is the zero-based trial index within this call.
   * With verbosity on and `progress_bar` unset or true, a progress bar over
   * the trials goes to stderr.
   *
   * @param integrand the integrand
   * @param ntrials number of trials, must be > 0
   * @param event_callback called as `event_callback(const point&, weight)`
   *        for each accepted event; may return an `EventSignal` to stop
   * @return statistics of the run
   * @throws std::invalid_argument if the grid is not frozen, ntrials == 0, or
   *         `progress_step` is invalid while the progress bar is shown
   * @throws std::runtime_error if the envelope is not ready or does not match
   *         the current grid, or f is non-finite in strict mode
   * @throws std::overflow_error if the weights or their sums overflow
   */
  template <typename I, typename ECB>
  gen_result_type generate_trials(I&& integrand, U ntrials, ECB&& event_callback) {
    if (ntrials == U(0)) throw std::invalid_argument("generate_trials requires ntrials > 0");
    require_frozen();
    require_envelope_ready();

    point_type point{derived().ndim(), derived().user_data()};

    gen_result_type res;
    res.envelope_volume_ = envelope_volume();
    U ntrials_done = 0;
    // overweights are accumulated one by one, unit weights in bulk after the loop
    U n_negative_overweight = 0;

    std::optional<util::ProgressBar> bar;
    U milestone_step = 0;
    U next_milestone = ntrials;  // never reached inside the loop without a bar
    const auto update_bar = [&] {
      bar->update(static_cast<double>(ntrials_done) / static_cast<double>(ntrials),
                  std::format("trials {}/{}", ntrials_done, ntrials));
    };
    const auto& opts = derived().opts_;
    if (opts.progress_bar.value_or(true) && opts.verbosity.value_or(0) > 0) {
      const double progress_step = opts.progress_step.value_or(DEFAULT_PROGRESS_STEP);
      if (!is_valid_progress_step(progress_step)) {
        throw std::invalid_argument("progress_step must be > 0 and <= 1");
      }
      milestone_step =
          util::math::max(U(1), static_cast<U>(static_cast<double>(ntrials) * progress_step));
      next_milestone = milestone_step;
      bar.emplace();
    }

    while (ntrials_done < ntrials) {
      // update progress up here so rejected and non-finite trials count too
      if (ntrials_done >= next_milestone) [[unlikely]] {
        next_milestone += milestone_step;
        update_bar();
      }
      const T envelope_value = derived().env_propose(point);
      // check R before a non-finite integrand can skip the trial
      if (!(envelope_value > T(0)) || !std::isfinite(envelope_value)) {
        throw std::runtime_error("envelope value is not finite and positive");
      }
      point.sample_index = ntrials_done;
      ++ntrials_done;
      const T fval = point.weight * integrand(point);
      if (!std::isfinite(fval)) {
        ++res.n_nonfinite_;
        if (strict_finite_integrand()) {
          throw std::runtime_error("non-finite integrand contribution");
        }
        continue;
      }
      const T abs_fval = util::math::abs(fval);
      T event_weight;
      if (abs_fval > envelope_value) {
        // overweight event: always accept and keep the factor |f|/R in the weight
        const T w_over = abs_fval / envelope_value;
        res.n_overweight_++;
        res.max_overweight_ = util::math::max(res.max_overweight_, w_over);
        event_weight = T(util::math::sgn(fval)) * w_over;
        accumulate_checked(res.acc_, event_weight);
        if (event_weight < T(0)) n_negative_overweight++;
      } else {
        // rejection sampling: accept with probability |f|/R
        const T r = derived().ran();
        if (r * envelope_value < abs_fval) {
          event_weight = T(util::math::sgn(fval));
        } else {
          continue;  // rejected: contributes 0, counted after the loop
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

    // unit weights sum to (#positive - #negative), their squares to n_unit
    const U n_unit = res.n_events_ - res.n_overweight_;
    const U n_unit_negative = res.n_negative_ - n_negative_overweight;
    int_acc_type unit_acc;
    unit_acc.reset(T(n_unit - n_unit_negative) - T(n_unit_negative), T(n_unit), n_unit);
    res.acc_.accumulate(unit_acc);
    // rejected trials add 0 to the sums, only the count changes
    res.acc_.accumulate_zeros(ntrials_done - res.n_events_);
    require_finite_sums(res.acc_);
    return res;
  }

  /// @}

  /// @name Envelope Queries & Manipulation
  /// @{

  /// @brief whether the integration grid is frozen (required for generation)
  [[nodiscard]] inline bool is_frozen() const {
    return derived().opts_.frozen.value_or(false);
  }

  /// @brief identity of the sampling map the envelope factors belong to;
  ///        overridden by generators whose map isn't fixed by the grid alone
  [[nodiscard]] inline util::HashValue_t env_grid_hash() const {
    return derived().hash().value();
  }

  /// @brief whether envelope and proposal caches match the current sampling map;
  ///        a changed map needs re-initialization or a compatible import
  [[nodiscard]] inline bool envelope_ready() const {
    return envelope_ready_ && envelope_grid_hash_ == derived().env_grid_hash();
  }

  /// @brief the local initialization seed, or zero if there is none
  ///        (e.g. envelope adopted from a `.khe` file)
  [[nodiscard]] inline T envelope_seed() const noexcept {
    return envelope_seed_;
  }

  /*!
   * @brief running estimate of the absolute integral A
   *
   * Includes initialization samples, raising passes, collected samples and
   * imported batch statistics. Stays empty when initialized from a constant
   * seed.
   */
  [[nodiscard]] inline const int_acc_type& abs_integral_estimate() const noexcept {
    return abs_acc_;
  }

  /// @brief envelope volume V = int R(u) du; events of a generation run are
  ///        normalized with V / n_trials
  [[nodiscard]] inline T envelope_volume() const noexcept {
    return envelope_volume_;
  }

  /*!
   * @brief predicted unweighting efficiency A / V (accepted events per trial)
   *
   * Uses `abs_integral_estimate()` if it has samples, otherwise the
   * initialization seed. The actual efficiency is int min(|f|, R) du / V,
   * which is at most A / V up to the statistical error on A. Not clamped
   * to one.
   *
   * @throws std::runtime_error if the envelope is not ready or does not match
   *         the current grid, or if the A estimate is not finite and positive
   *         (importing factors alone gives neither A nor a seed)
   */
  [[nodiscard]] T predicted_efficiency() const {
    require_envelope_ready();
    const T a_est = abs_acc_.count() > U(0) ? abs_acc_.value() : envelope_seed_;
    if (!(a_est > T(0)) || !std::isfinite(a_est)) {
      throw std::runtime_error(
          "predicted_efficiency requires a finite positive abs-integral estimate");
    }
    return a_est / envelope_volume_;
  }

  /*!
   * @brief multiply the whole envelope by `factor` (e.g. a safety factor)
   *
   * Leaves the envelope unchanged on failure.
   *
   * @param factor scaling factor, must be finite and positive
   * @throws std::invalid_argument if the grid is not frozen or the factor is invalid
   * @throws std::runtime_error if the envelope is not initialized or does not
   *         match the current grid, if factor < 1 with pending data, or if
   *         the scaled proposal can't be built
   */
  inline void scale_envelope(T factor) {
    require_frozen();
    if (!(factor > T(0)) || !std::isfinite(factor)) {
      throw std::invalid_argument("scale_envelope requires a finite positive factor");
    }
    // don't bind factors from an old sampling map to the current one
    require_envelope_ready();
    if (factor < T(1)) require_no_envelope_data();
    // spread the scale evenly over the ndim factors of R
    const T factor_scale = std::pow(factor, T(1) / T(derived().ndim()));
    ndarray::NDArray<T, S> candidate(envelope_.shape());
    std::ranges::transform(envelope_, candidate.begin(),
                           [factor_scale](T value) { return value * factor_scale; });
    seal_envelope(candidate, derived().env_grid_hash());
  }

  /// @}

  /// @name Envelope Files
  /// @{

  /*!
   * @brief save the envelope and pending observations to a `.khe` file
   *
   * The version-1 `.khe` payload holds the sampling-map hash, the factors and
   * the pending batch, if any. The batch stays pending. Use `save()` for a
   * full `.khs` checkpoint.
   *
   * @param filepath path of the `.khe` file
   * @throws std::invalid_argument if the grid is not frozen
   * @throws std::runtime_error if the envelope is not initialized or stale
   */
  void save_envelope(const std::filesystem::path& filepath) const {
    require_frozen();
    require_envelope_ready();
    this->write_file(filepath, detail::FileType::ENVELOPE, [this](std::ostream& out) {
      using namespace kakuhen::util::serialize;
      serialize_one<uint8_t>(out, envelope_file_version);
      serialize_one<util::HashValue_t>(out, envelope_grid_hash_);
      envelope_.serialize(out);
      envelope_data_.serialize(out, derived().ndim());
    });
  }

  /// @brief default `.khe` path (same naming as `.khd`)
  template <typename D = Derived>
  [[nodiscard]] std::filesystem::path file_envelope() const
    requires detail::HasPrefix<D>
  {
    auto filepath = this->file_data();
    filepath.replace_extension(detail::suffix_envelope);
    return filepath;
  }

  /// @brief save to the default `.khe` path (`file_envelope()`)
  /// @return the saved path
  template <typename D = Derived>
  std::filesystem::path save_envelope() const
    requires detail::HasPrefix<D>
  {
    auto filepath = file_envelope();
    save_envelope(filepath);
    return filepath;
  }

  /*!
   * @brief merge envelope factors and process observations from a `.khe` file
   *
   * If the local envelope is ready, takes the maximum of corresponding
   * factors. Otherwise replaces it and sets the seed to zero, which requires
   * that no local data are pending. The sampling map must match, including
   * BASIN's sampling order. It's up to the caller to make sure both runs used
   * the same integrand and parameters; RNG seeds, training budgets and
   * initial envelopes may differ.
   *
   * Incoming observations are processed in file order, at most one raise
   * each, and their batch statistics are added to the A estimate. Local
   * pending data are left alone. Import order matters when observations are
   * present, and importing a batch twice counts it twice. Merging factors
   * alone is idempotent.
   *
   * On failure the generator is unchanged, proposal caches included.
   *
   * @param filepath path of the `.khe` file
   * @return merged A estimate and volume, plus the counts of the incoming
   *         batch: evaluations, records (`n_violations`) and records that
   *         raised the envelope (all zero without a batch)
   * @throws std::invalid_argument if the grid is not frozen
   * @throws std::runtime_error if the file or envelope is invalid or
   *         incompatible, a pending batch belongs to an envelope that is not
   *         ready, or the merged proposal has no finite volume
   * @throws std::overflow_error if the merged statistics overflow
   */
  env_result_type merge_envelope(const std::filesystem::path& filepath) {
    require_frozen();
    const bool merge_existing = envelope_ready();
    if (!merge_existing) require_no_envelope_data();

    ndarray::NDArray<T, S> merged;
    envelope_data_type incoming;
    this->read_file(filepath, detail::FileType::ENVELOPE, [&](std::istream& in) {
      using namespace kakuhen::util::serialize;
      uint8_t version;
      deserialize_one<uint8_t>(in, version);
      if (version != envelope_file_version) {
        throw std::runtime_error("unsupported envelope file version");
      }
      util::HashValue_t grid_hash;
      deserialize_one<util::HashValue_t>(in, grid_hash);
      if (grid_hash != derived().env_grid_hash()) {
        throw std::runtime_error("envelope file does not match the sampling map");
      }
      merged.deserialize_expected_shape(in, derived().env_shape());
      if (!std::ranges::all_of(merged,
                               [](T factor) { return factor > T(0) && std::isfinite(factor); })) {
        throw std::runtime_error("envelope file factors must be finite and positive");
      }
      incoming = envelope_data_type::deserialize(in, derived().ndim(), merged.size());
    });
    if (merge_existing) {
      std::ranges::transform(merged, envelope_, merged.begin(),
                             [](T a, T b) { return std::max(a, b); });
    }

    auto merged_acc = abs_acc_;
    U n_raised = 0;
    if (incoming.present()) {
      require_count_capacity(incoming.acc.count());
      merged_acc.accumulate(incoming.acc);
      require_finite_sums(merged_acc);
      n_raised = apply_envelope_records(merged, incoming);
    }
    seal_envelope(merged, derived().env_grid_hash());
    abs_acc_ = merged_acc;
    if (!merge_existing) envelope_seed_ = T(0);
    return {.abs_acc_ = abs_acc_,
            .n_violations_ = static_cast<U>(incoming.values.size()),
            .n_raised_ = n_raised,
            .envelope_volume_ = envelope_volume(),
            .n_evaluations_ = incoming.acc.count()};
  }

  /// @brief merge from the default `.khe` path (`file_envelope()`)
  template <typename D = Derived>
  env_result_type merge_envelope()
    requires detail::HasPrefix<D>
  {
    return merge_envelope(file_envelope());
  }

  /// @}

  /// @name State Persistence
  /// @{

  /*!
   * @brief save integrator state, envelope and pending batch to a `.khs` file
   *
   * The envelope block is appended to the integrator state, so the plain
   * integrator can still read the grid from this file.
   *
   * @param filepath path of the state file
   */
  void save(const std::filesystem::path& filepath) const {
    this->write_file(filepath, detail::FileType::STATE,
                     [this](std::ostream& out) { write_state_stream(out); });
  }

  /// @brief save the generator state to the default state file
  /// @return the saved path
  std::filesystem::path save() const {
    std::filesystem::path fstate = this->file_state();
    save(fstate);
    return fstate;
  }

  /*!
   * @brief load integrator state and envelope from a `.khs` file
   *
   * A file without an envelope block leaves the envelope uninitialized.
   * Pending data are only kept if the envelope was ready when saved and
   * matches the loaded map.
   *
   * @param filepath path of the state file
   */
  void load(const std::filesystem::path& filepath) {
    if (!this->state_file_exists(filepath)) return;
    this->read_file(filepath, detail::FileType::STATE,
                    [this](std::istream& in) { read_state_stream(in); });
  }

  /// @brief load the generator state from the default state file
  /// @return the loaded path
  std::filesystem::path load() {
    std::filesystem::path fstate = this->file_state();
    load(fstate);
    return fstate;
  }

  /// @}

  /// @name State Streams (integrator state + envelope block)
  /// @{

  /// @brief write the integrator state followed by the envelope block
  void write_state_stream(std::ostream& out) const {
    IntBase::write_state_stream(out);
    write_envelope_stream(out);
  }

  /// @brief read the integrator state and, if present, the envelope block
  void read_state_stream(std::istream& in) {
    // reset first so a failed read does not leave a ready envelope behind
    reset_envelope_state();
    IntBase::read_state_stream(in);
    read_envelope_stream(in);
  }

  /// @}

 protected:
  /// @brief access the derived generator
  inline Derived& derived() noexcept {
    return static_cast<Derived&>(*this);
  }

  /// @brief access the derived generator
  inline const Derived& derived() const noexcept {
    return static_cast<const Derived&>(*this);
  }

 private:
  using envelope_data_type = detail::EnvelopeData<T, U, S>;

  /// @brief throw if envelope data are pending
  void require_no_envelope_data() const {
    if (!has_envelope_data()) return;
    // a stale envelope can't be adapted, so its batch can only be cleared
    throw std::runtime_error(envelope_ready()
                                 ? "adapt or clear pending envelope data first"
                                 : "the envelope is not ready for the current sampling map; "
                                   "clear its pending data first");
  }

  /// @brief per-factor raise on a violation: 1 + 1 / (10 * ndim)
  [[nodiscard]] T envelope_raising_factor() const noexcept {
    return T(1) + T(1) / (T(10) * T(derived().ndim()));
  }

  /// @brief apply observations in order, at most one raise each;
  ///        observations and proposal caches are left untouched
  /// @return number of raises
  U apply_envelope_records(ndarray::NDArray<T, S>& table, const envelope_data_type& data) const {
    U n_raised = 0;
    const auto ndim = derived().ndim();
    const T gamma = envelope_raising_factor();
    for (std::size_t i = 0; i < data.values.size(); ++i) {
      if (detail::raise_envelope_record(table, data.cells(i, ndim), data.values[i], gamma)) {
        ++n_raised;
      }
    }
    return n_raised;
  }

  /// @brief throw if the integration grid is not frozen
  void require_frozen() const {
    if (!is_frozen()) {
      throw std::invalid_argument("a frozen integration grid is required");
    }
  }

  /// @brief throw unless proposal caches are valid and match the sampling map
  void require_envelope_ready() const {
    if (!envelope_ready_) {
      throw std::runtime_error("an initialized or imported envelope is required");
    }
    if (envelope_grid_hash_ != derived().env_grid_hash()) {
      throw std::runtime_error(
          has_envelope_data()
              ? "envelope does not match the current grid; call clear_envelope_data() before "
                "re-initializing"
              : "envelope does not match the current grid; re-initialize");
    }
  }

  /*!
   * @brief evaluate |f| at `neval` points sampled from the grid
   *
   * Calls `visit(abs_fval, finite, indices)` for each sample, with the
   * envelope factor indices at the point; a non-finite sample has
   * `abs_fval = 0`. Stops when `visit` returns false.
   *
   * The caller must make sure the sample count fits before changing state
   * (initialization starts from an empty accumulator, so it's fine there).
   */
  template <typename I, typename V>
  void for_each_abs_sample(I& integrand, U neval, V&& visit) {
    const S ndim = derived().ndim();
    point_type point{ndim, derived().user_data()};
    std::vector<T> u_buf(ndim);
    std::vector<S> indices(ndim);
    auto cell = derived().make_cell_ctx();
    for (U i = 0; i < neval; ++i) {
      for (T& u : u_buf)
        u = derived().ran();
      point.sample_index = i;
      derived().map_point(u_buf, point, cell);
      T abs_fval;
      const bool finite = eval_abs(integrand, point, abs_fval);
      derived().env_indices(cell, indices);
      if (!visit(abs_fval, finite, std::span<const S>(indices))) break;
    }
  }

  /// @brief throw if `neval` more evaluations would overflow the A count;
  ///        the pending batch is a subset of `abs_acc_`, so this covers its count too
  void require_count_capacity(U neval) const {
    if (neval > std::numeric_limits<U>::max() - abs_acc_.count()) {
      throw std::overflow_error("evaluation count overflows");
    }
  }

  /// @brief count one evaluation in `result`
  static void count_evaluation(env_result_type& result, bool finite) noexcept {
    ++result.n_evaluations_;
    if (!finite) ++result.n_nonfinite_;
  }

  /// @brief evaluate |f| at `point` into `abs_fval`
  /// @return whether f is finite; if not, throws in strict mode or sets
  ///         `abs_fval` to zero
  template <typename I>
  bool eval_abs(I& integrand, point_type& point, T& abs_fval) const {
    const T fval = point.weight * integrand(point);
    if (!std::isfinite(fval)) {
      if (strict_finite_integrand()) {
        throw std::runtime_error("non-finite integrand contribution");
      }
      abs_fval = T(0);
      return false;
    }
    abs_fval = util::math::abs(fval);
    return true;
  }

  /// @brief whether a non-finite integrand throws instead of counting as zero
  [[nodiscard]] bool strict_finite_integrand() const noexcept {
    return derived().opts_.strict_finite_integrand.value_or(false);
  }

  /// @brief square `value`, throw if the result is not finite
  static T checked_square(T value) {
    const T square = value * value;
    if (!std::isfinite(square)) {  // also catches a non-finite value
      throw std::overflow_error(
          "contribution or square overflows; rebuild the envelope "
          "from an absolute-integral estimate or rescale the integrand");
    }
    return square;
  }

  /// @brief accumulate a value with finite square; caller must check the sums
  static void accumulate_checked(int_acc_type& acc, T value) {
    acc.accumulate(value, checked_square(value));
  }

  /// @brief throw if either accumulated sum is not finite
  static void require_finite_sums(const int_acc_type& acc) {
    if (!acc.is_finite()) {
      throw std::overflow_error("accumulated contributions overflow");
    }
  }

  /// @brief reset readiness, statistics and pending data before initializing or loading
  void reset_envelope_state() {
    envelope_ready_ = false;
    envelope_grid_hash_ = {};
    envelope_seed_ = T(0);
    envelope_volume_ = T(0);
    abs_acc_.reset();
    clear_envelope_data();
  }

  /// @brief seed a flat envelope R(u) = seed and seal it
  void seed_envelope(T seed) {
    if (!(seed > T(0)) || !std::isfinite(seed)) {
      throw std::invalid_argument("initialize_envelope requires a finite seed > 0");
    }
    // equal factors whose product is the seed
    const T factor = std::pow(seed, T(1) / T(derived().ndim()));
    envelope_ = ndarray::NDArray<T, S>(derived().env_shape());
    envelope_.fill(factor);
    envelope_seed_ = seed;
    seal_envelope();
  }

  /*!
   * @brief build proposal caches, install the table and mark the envelope ready
   *
   * Binds the envelope to `grid_hash` and moves `table` into `envelope_`
   * unless they're the same object. Caches and volume are built and checked
   * before any state changes.
   */
  void seal_envelope(ndarray::NDArray<T, S>& table, kakuhen::util::HashValue_t grid_hash) {
    auto [volume, cache] = derived().env_prepare(table);
    if (!(volume > T(0)) || !std::isfinite(volume)) {
      throw std::runtime_error("generation envelope has invalid volume");
    }
    static_assert(noexcept(derived().env_commit_cache(std::move(cache))));
    if (&table != &envelope_) envelope_ = std::move(table);
    derived().env_commit_cache(std::move(cache));
    envelope_volume_ = volume;
    envelope_grid_hash_ = grid_hash;
    envelope_ready_ = true;
  }

  /// @brief seal the envelope and bind it to the current sampling map
  void seal_envelope() {
    seal_envelope(envelope_, derived().env_grid_hash());
  }

  /*!
   * @brief append the envelope block to a state stream
   *
   * Writes tag, version, ready flag, seed, map hash, A accumulator, factor
   * table and pending batch. Caches and volume are rebuilt on loading.
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
    envelope_data_.serialize(out, derived().ndim());
  }

  /*!
   * @brief read an optional envelope block after the integrator state
   *
   * `read_state_stream` must reset the envelope first. EOF or a trailing
   * user-data record means no envelope was saved (the record is left unread).
   * The stored map hash is kept, so loading can't make a stale envelope
   * ready. Pending data are only kept with an envelope ready for the loaded
   * map.
   *
   * @throws std::runtime_error if the envelope block is corrupt or has an
   *         unsupported version
   */
  void read_envelope_stream(std::istream& in) {
    using namespace kakuhen::util::serialize;
    // dispatch on the first byte, works on streams that can't seek too
    const auto next = in.peek();
    if (next == std::istream::traits_type::eof() || next == util::USER_DATA_HEADER.front()) {
      if (next != std::istream::traits_type::eof()) require_user_data_header(in);
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
    auto data = envelope_data_type::deserialize(in, derived().ndim(), envelope_.size());
    if (data.acc.count() > abs_acc_.count()) {
      throw std::runtime_error("envelope batch exceeds cumulative evaluation count");
    }
    if (ready != 0) seal_envelope(envelope_, grid_hash);
    if (envelope_ready()) {
      envelope_data_ = std::move(data);
    } else if (data.present()) {
      derived().print_info_message("state", "envelope not ready; pending batch dropped");
    }
  }

  /// @brief check the user-data tag without advancing a seekable stream;
  ///        on non-seekable streams the user-data reader does the check
  static void require_user_data_header(std::istream& in) {
    const auto start = in.tellg();
    if (start == std::streampos(-1)) return;
    std::array<char, util::USER_DATA_HEADER.size()> marker{};
    in.read(marker.data(), static_cast<std::streamsize>(marker.size()));
    const bool found =
        in && std::string_view(marker.data(), marker.size()) == util::USER_DATA_HEADER;
    in.clear();
    in.seekg(start);
    if (!found || !in) throw std::runtime_error("invalid envelope block tag in state stream");
  }

 protected:
  /// @brief write the expected shape and factor table, or zeros if the shapes differ
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

  /// @brief read the factor table and check its shape against the loaded integrator
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
    // NDArray stores its own shape; check it against the dimensions read above
    if (!std::ranges::equal(envelope_.shape(), shape)) {
      throw std::runtime_error("corrupt envelope block: unexpected table shape");
    }
  }

  /// raw envelope table; the `ndim` factors at a point multiply to R(u)
  ndarray::NDArray<T, S> envelope_;

 private:
  /// tag of the envelope block at the end of `.khs` state streams
  static constexpr std::string_view envelope_block_tag = "KHENVB";  // 6 bytes
  static_assert(envelope_block_tag.front() != util::USER_DATA_HEADER.front(),
                "the envelope block and user-data records are told apart by their first byte");
  /// version of the envelope block
  static constexpr uint8_t envelope_block_version = 1;
  /// version of the standalone `.khe` payload
  static constexpr uint8_t envelope_file_version = 1;

  /// running estimate of the absolute integral A
  int_acc_type abs_acc_;
  /// observations and batch statistics since the last adapt or clear
  envelope_data_type envelope_data_;
  /// seed of the flat initial envelope (zero if none)
  T envelope_seed_{};
  /// volume of the sealed envelope V = int R(u) du
  T envelope_volume_{};
  /// whether the proposal caches are built for `envelope_`
  bool envelope_ready_ = false;
  /// hash of the sampling map the envelope belongs to
  kakuhen::util::HashValue_t envelope_grid_hash_{};
};

}  // namespace kakuhen::integrator
