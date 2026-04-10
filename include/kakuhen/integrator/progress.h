#pragma once

#include <cstdint>
#include <type_traits>

namespace kakuhen::integrator {

/// @brief Types of progress events fired during integration.
enum class ProgressEventKind : uint8_t {
  START,           ///< Fired before the first iteration begins.
  ITER_START,      ///< Fired before each iteration begins.
  EVAL_MILESTONE,  ///< Fired when per-iteration progress reaches a milestone (e.g., 25%, 50% of neval).
  ITER_END,        ///< Fired after each iteration completes.
  END              ///< Fired after all iterations complete (or on cancellation/exception).
};

/// @brief Signal returned by progress callbacks to control integration flow.
enum class EventSignal : uint8_t {
  NONE = 0,       ///< Continue normally.
  CANCEL = 1,     ///< Stop integration gracefully, return partial results.
  EXCEPTION = 2,  ///< Indicates an exception occurred in the callback.
};

/// @brief Bitwise OR for EventSignal.
constexpr EventSignal operator|(EventSignal a, EventSignal b) noexcept {
  using U = std::underlying_type_t<EventSignal>;
  return static_cast<EventSignal>(static_cast<U>(a) | static_cast<U>(b));
}

/// @brief Bitwise OR assignment for EventSignal.
constexpr EventSignal& operator|=(EventSignal& a, EventSignal b) noexcept {
  return a = a | b;
}

/// @brief Bitwise AND for EventSignal.
constexpr EventSignal operator&(EventSignal a, EventSignal b) noexcept {
  using U = std::underlying_type_t<EventSignal>;
  return static_cast<EventSignal>(static_cast<U>(a) & static_cast<U>(b));
}

/// @brief Check if a signal flag is set.
[[nodiscard]] constexpr bool has_signal(EventSignal flags, EventSignal test) noexcept {
  using U = std::underlying_type_t<EventSignal>;
  return (static_cast<U>(flags) & static_cast<U>(test)) != 0;
}

/// @brief Default fraction of a single iteration's evaluations between progress milestone callbacks (5%).
inline constexpr double DEFAULT_PROGRESS_STEP = 0.05;

/// @brief True when `Cb` is a real progress callback (not `std::nullptr_t`).
template <typename Cb>
inline constexpr bool is_progress_callback_v =
    !std::is_same_v<std::remove_cvref_t<Cb>, std::nullptr_t>;

/*!
 * @brief Event data passed to progress callbacks.
 *
 * Contains information about the current state of integration when a
 * progress event is fired.
 *
 * @note For `EVAL_MILESTONE` events fired mid-iteration, `value` and `error`
 * reflect the accumulated result from *completed* iterations only, not the
 * current partial iteration. For `ITER_END` events, they include the just-
 * completed iteration.
 *
 * @tparam T The value type for integral results (e.g., double).
 * @tparam U The count type for evaluation counts (e.g., uint64_t).
 */
template <typename T, typename U>
struct ProgressEvent {
  ProgressEventKind kind;  ///< Type of event.
  U niter;                 ///< Total number of iterations.
  U current_iter;          ///< Current iteration (0-indexed).
  U neval;                 ///< Number of evaluations per iteration.
  U current_eval;          ///< Evaluations completed so far in the current iteration (1-indexed).
  T value;                 ///< Current integral estimate (from completed iterations).
  T error;                 ///< Current error estimate (from completed iterations).
  double fraction;         ///< Per-iteration progress fraction: current_eval / neval, in [0.0, 1.0].
                           ///< Resets to 0 at each ITER_START and reaches 1.0 at ITER_END.
  double elapsed_start;    ///< Elapsed time in seconds since integration started.
  double elapsed_iter;     ///< Elapsed time in seconds since current iteration started.
};

}  // namespace kakuhen::integrator
