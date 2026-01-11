#pragma once

#include "kakuhen/util/accumulator.h"

namespace kakuhen::integrator {

/// @todo Update with algo in https://arxiv.org/pdf/2206.10662
/// @todo Also write an accum function add(f) and add(f,f2)

/*!
 * @brief Accumulates integral values and their squares for statistical analysis.
 *
 * This struct is used to accumulate function values and their squares during
 * Monte Carlo integration. It provides methods to calculate the mean,
 * variance, and error of the integral.
 *
 * @tparam T The value type for the integral (e.g., double).
 * @tparam U The count type for the number of evaluations (e.g., uint64_t).
 */
template <typename T, typename U>
struct IntegralAccumulator {
  using value_type = T;
  using count_type = U;

  using acc_type = kakuhen::util::accumulator::Accumulator<T>;

  acc_type f_{};   //!< Accumulator for the sum of function values.
  acc_type f2_{};  //!< Accumulator for the sum of squared function values.
  U n_ = 0;        //!< Number of accumulated samples.

  /// @name State Manipulation
  /// @{

  /*!
   * @brief Accumulate a single function value.
   *
   * Adds `f` to the sum of function values and `f*f` to the sum of squared
   * function values. Increments the sample count.
   *
   * @param f The function value to accumulate.
   */
  inline void accumulate(const T& f) noexcept {
    f_.add(f);
    f2_.add(f * f);
    n_++;
  }

  /*!
   * @brief Accumulate a function value and its square.
   *
   * Adds `f` to the sum of function values and `f2` to the sum of squared
   * function values. Increments the sample count.
   *
   * @param f The function value to accumulate.
   * @param f2 The squared function value to accumulate.
   */
  inline void accumulate(const T& f, const T& f2) noexcept {
    f_.add(f);
    f2_.add(f2);
    n_++;
  }

  /*!
   * @brief Accumulate values from another IntegralAccumulator.
   *
   * Adds the accumulated sums and counts from another `IntegralAccumulator`
   * object to this one.
   *
   * @param other The other `IntegralAccumulator` to accumulate from.
   */
  inline void accumulate(const IntegralAccumulator<T, U>& other) noexcept {
    f_.add(other.f_.result());
    f2_.add(other.f2_.result());
    n_ += other.n_;
  }

  /*!
   * @brief Resets the accumulator to its initial state (zero sums, zero count).
   */
  inline void reset() noexcept {
    f_.reset();
    f2_.reset();
    n_ = 0;
  }

  /*!
   * @brief Resets the accumulator with specific pre-calculated values.
   *
   * @param f The new sum of function values.
   * @param f2 The new sum of squared function values.
   * @param n The new number of accumulated samples.
   */
  inline void reset(const T& f, const T& f2, const U& n) noexcept {
    f_.reset(f);
    f2_.reset(f2);
    n_ = n;
  }

  /// @}

  /// @name Statistics
  /// @{

  /*!
   * @brief Get the total number of accumulated samples.
   *
   * @return The number of samples.
   */
  [[nodiscard]] inline U count() const noexcept {
    return n_;
  }

  /*!
   * @brief Calculate the mean value of the accumulated function values.
   *
   * @return The mean value.
   */
  [[nodiscard]] inline T value() const noexcept {
    return f_.result() / T(n_);
  }

  /*!
   * @brief Calculate the variance of the mean value.
   *
   * This is the variance of the expectation value, not the variance of the
   * underlying distribution.
   *
   * @return The variance of the mean value. Returns 0 if `n_ <= 1`.
   */
  [[nodiscard]] inline T variance() const noexcept {
    if (n_ > 1) {
      return (f2_.result() / T(n_) - value() * value()) / T(n_ - 1);
    } else {
      return T(0);
    }
  }

  /*!
   * @brief Calculate the standard deviation (error) of the mean value.
   *
   * @return The standard deviation of the mean value.
   */
  [[nodiscard]] inline T error() const noexcept {
    return std::sqrt(variance());
  }

  /// @}

  /// @name Serialization
  /// @{

  /*!
   * @brief Serializes the accumulator to an output stream.
   *
   * @param out The output stream.
   * @param with_type Whether to include type information.
   */
  void serialize(std::ostream& out, bool with_type = false) const noexcept {
    if (with_type) {
      int16_t T_tos = kakuhen::util::type::get_type_or_size<T>();
      int16_t U_tos = kakuhen::util::type::get_type_or_size<U>();
      kakuhen::util::serialize::serialize_one<int16_t>(out, T_tos);
      kakuhen::util::serialize::serialize_one<int16_t>(out, U_tos);
    }
    kakuhen::util::serialize::serialize_one<T>(out, f_.result());
    kakuhen::util::serialize::serialize_one<T>(out, f2_.result());
    kakuhen::util::serialize::serialize_one<U>(out, n_);
  }

  /*!
   * @brief Deserializes the accumulator from an input stream.
   *
   * @param in The input stream.
   * @param with_type Whether to verify type information.
   * @throws std::runtime_error if type information mismatches.
   */
  void deserialize(std::istream& in, bool with_type = false) {
    if (with_type) {
      int16_t T_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, T_tos);
      if (T_tos != kakuhen::util::type::get_type_or_size<T>()) {
        throw std::runtime_error("type or size mismatch for typename T");
      }
      int16_t U_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, U_tos);
      if (U_tos != kakuhen::util::type::get_type_or_size<U>()) {
        throw std::runtime_error("type or size mismatch for typename U");
      }
    }
    T value;
    kakuhen::util::serialize::deserialize_one<T>(in, value);
    f_.reset(value);
    kakuhen::util::serialize::deserialize_one<T>(in, value);
    f2_.reset(value);
    kakuhen::util::serialize::deserialize_one<U>(in, n_);
  }

  /// @}

};  // struct IntegralAccumulator

/*!
 * @brief Factory function to create an IntegralAccumulator from value, error, and count.
 *
 * This function reconstructs an `IntegralAccumulator` object given a mean
 * value, its error, and the number of samples. This is useful for initializing
 * an accumulator from summary statistics.
 *
 * @tparam T The value type.
 * @tparam U The count type.
 * @param value The mean value of the integral.
 * @param error The error (standard deviation) of the integral.
 * @param n The number of samples.
 * @return An `IntegralAccumulator` object initialized with the provided data.
 */
template <typename T, typename U>
inline IntegralAccumulator<T, U> make_integral_accumulator(const T& value, const T& error,
                                                           const U& n) {
  T f = value * T(n);
  T f2;
  if (n > 1) {
    f2 = T(n) * (value * value + T(n - 1) * error * error);
  } else {
    f2 = f * f;  // if n==1, variance is zero
  }
  IntegralAccumulator<T, U> acc{};
  acc.f_.reset(f);
  acc.f2_.reset(f2);
  acc.n_ = n;
  return acc;
}

}  // namespace kakuhen::integrator
