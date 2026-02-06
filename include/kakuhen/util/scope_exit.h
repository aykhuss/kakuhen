#pragma once

#include <type_traits>
#include <utility>

namespace kakuhen::util {

/**
 * @brief RAII class that executes a function when it goes out of scope.
 *
 * This utility is useful for ensuring that certain clean-up or restoration
 * logic is performed regardless of how a block is exited (normal return,
 * exception, etc.).
 *
 * @tparam F The type of the callable to execute.
 */
template <typename F>
class ScopeExit {
 public:
  /**
   * @brief Constructs a ScopeExit object with a given callable.
   * @param f The callable to execute on destruction.
   */
  explicit ScopeExit(F&& f) noexcept(std::is_nothrow_move_constructible_v<F>)
      : f_(std::move(f)), active_(true) {}

  /**
   * @brief Move constructor. Transfers the callable and takes over responsibility.
   * @param other The other ScopeExit object.
   */
  ScopeExit(ScopeExit&& other) noexcept(std::is_nothrow_move_constructible_v<F>)
      : f_(std::move(other.f_)), active_(other.active_) {
    other.active_ = false;
  }

  // Prevent copying to avoid multiple executions of the same cleanup logic.
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

  /**
   * @brief Destructor. Executes the callable if it has not been released.
   */
  ~ScopeExit() noexcept {
    if (active_) {
      f_();
    }
  }

  /**
   * @brief Prevents the callable from being executed when the object is destroyed.
   */
  void release() noexcept { active_ = false; }

 private:
  F f_;          //!< The cleanup function.
  bool active_;  //!< Whether the cleanup function should be executed.
};

/**
 * @brief Factory function to create a ScopeExit object.
 *
 * This function uses template argument deduction to simplify the creation
 * of scope guards.
 *
 * @tparam F The callable type.
 * @param f The callable to execute at the end of the scope.
 * @return A ScopeExit object.
 */
template <typename F>
[[nodiscard]] ScopeExit<std::decay_t<F>> scope_exit(F&& f) {
  return ScopeExit<std::decay_t<F>>(std::forward<F>(f));
}

/**
 * @brief Factory function to create a ScopeExit object, aliased as 'defer'.
 *
 * This is an alias for `scope_exit` that mimics the 'defer' keyword found in
 * other languages like Go or Swift.
 *
 * @tparam F The callable type.
 * @param f The callable to execute at the end of the scope.
 * @return A ScopeExit object.
 */
template <typename F>
[[nodiscard]] ScopeExit<std::decay_t<F>> defer(F&& f) {
  return scope_exit(std::forward<F>(f));
}

}  // namespace kakuhen::util
