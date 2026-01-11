#pragma once

#include "kakuhen/util/serialize.h"
#include "kakuhen/util/type.h"

namespace kakuhen::integrator {

/// @todo Should maybe consider DOD design for better cache alignment?

/*!
 * @brief Accumulates values for a grid cell.
 *
 * This struct is used to accumulate a sum of values and a count of contributions
 * for a specific cell within a grid, typically used in adaptive integration
 * algorithms like VEGAS to store information about the integrand's behavior
 * in different regions.
 *
 * @tparam T The value type to accumulate (e.g., double).
 * @tparam U The count type for the number of accumulations (e.g., uint64_t).
 */
template <typename T, typename U>
struct GridAccumulator {
  using value_type = T;
  using count_type = U;

  T acc_ = T(0);  //!< The accumulated value.
  U n_ = U(0);    //!< The number of times a value has been accumulated.

  /// @name State Manipulation
  /// @{

  /*!
   * @brief Accumulate a single value.
   *
   * Adds the given value to the accumulator and increments the count.
   *
   * @param x The value to accumulate.
   */
  inline void accumulate(const T& x) noexcept {
    acc_ += x;
    n_++;
  }

  /*!
   * @brief Accumulate values from another GridAccumulator.
   *
   * Adds the accumulated value and count from another `GridAccumulator`
   * object to this one.
   *
   * @param other The other `GridAccumulator` to accumulate from.
   */
  inline void accumulate(const GridAccumulator<T, U>& other) noexcept {
    acc_ += other.acc_;
    n_ += other.n_;
  }

  /*!
   * @brief Resets the accumulator to its initial state (zero value, zero count).
   */
  inline void reset() noexcept {
    acc_ = T(0);
    n_ = U(0);
  }

  /*!
   * @brief Resets the accumulator with specific pre-calculated values.
   *
   * @param acc The new accumulated value.
   * @param n The new number of accumulations.
   */
  inline void reset(const T& acc, const U& n) noexcept {
    acc_ = acc;
    n_ = n;
  }

  /// @}

  /// @name Operators
  /// @{

  /*!
   * @brief Adds a value to the accumulator.
   * @param x The value to add.
   * @return A reference to this accumulator.
   */
  inline GridAccumulator& operator+=(const T& x) noexcept {
    accumulate(x);
    return *this;
  }

  /*!
   * @brief Adds another accumulator's data to this one.
   * @param other The other accumulator.
   * @return A reference to this accumulator.
   */
  inline GridAccumulator& operator+=(const GridAccumulator<T, U>& other) noexcept {
    accumulate(other);
    return *this;
  }

  /// @}

  /// @name Queries
  /// @{

  /*!
   * @brief Get the total number of accumulated values.
   *
   * @return The count of accumulated values.
   */
  [[nodiscard]] inline U count() const noexcept {
    return n_;
  }

  /*!
   * @brief Get the accumulated value.
   *
   * @return The current accumulated sum.
   */
  [[nodiscard]] inline T value() const noexcept {
    return acc_;
  }

  /// @}

  /// @name Serialization
  /// @{

  /*!
   * @brief Serializes the accumulator to an output stream.
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
    kakuhen::util::serialize::serialize_one<T>(out, acc_);
    kakuhen::util::serialize::serialize_one<U>(out, n_);
  }

  /*!
   * @brief Deserializes the accumulator from an input stream.
   * @param in The input stream.
   * @param with_type Whether to verify type information.
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
    kakuhen::util::serialize::deserialize_one<T>(in, acc_);
    kakuhen::util::serialize::deserialize_one<U>(in, n_);
  }

  /// @}

};  // struct GridAccumulator

}  // namespace kakuhen::integrator
