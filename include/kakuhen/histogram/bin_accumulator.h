#pragma once

#include "kakuhen/util/accumulator.h"

namespace kakuhen::histogram {

/*!
 * @brief Accumulates statistics for a single histogram bin.
 *
 * This structure tracks the sum of weights ($W$) and the sum of squared weights
 * ($W^2$) for a single bin. It acts as the fundamental storage unit for histogram data.
 *
 * By default, it uses compensated summation (TwoSum) via the underlying
 * `Accumulator` type to ensure numerical stability even with large cancellations
 * in Monte Carlo weights. This behavior can be customized via the `Acc` template parameter.
 *
 * @tparam T The floating-point value type (default: double).
 * @tparam Acc The accumulator type to use (default: kakuhen::util::accumulator::Accumulator<T>).
 */
template <typename T = double, typename Acc = kakuhen::util::accumulator::Accumulator<T>>
struct BinAccumulator {
  using value_type = T;
  using acc_type = Acc;

  acc_type acc_wgt_{};     //!< Accumulator for the sum of weights ($W$).
  acc_type acc_wgt_sq_{};  //!< Accumulator for the sum of squared weights ($W^2$).

  /// @name Accumulation
  /// @{

  /*!
   * @brief Accumulate a weight into the bin.
   *
   * Updates the weight sum with `w` and the squared weight sum with `w*w`.
   *
   * @param w The weight to accumulate.
   */
  inline void accumulate(const T& w) noexcept {
    acc_wgt_.add(w);
    acc_wgt_sq_.add(w * w);
  }

  /*!
   * @brief Accumulate explicit weight and squared weight values.
   *
   * @param w The weight to add to the weight sum.
   * @param w2 The squared weight to add to the squared weight sum.
   */
  inline void accumulate(const T& w, const T& w2) noexcept {
    acc_wgt_.add(w);
    acc_wgt_sq_.add(w2);
  }

  /*!
   * @brief Accumulate another BinAccumulator into this one.
   *
   * Merges the statistics from `other` into this bin.
   *
   * @param other The other bin accumulator.
   */
  inline void accumulate(const BinAccumulator<T, Acc>& other) noexcept {
    acc_wgt_ += other.acc_wgt_;
    acc_wgt_sq_ += other.acc_wgt_sq_;
  }

  /*!
   * @brief Compound assignment operator for accumulating another bin.
   */
  inline BinAccumulator& operator+=(const BinAccumulator<T, Acc>& other) noexcept {
    accumulate(other);
    return *this;
  }

  /// @}

  /// @name Reset
  /// @{

  /*!
   * @brief Resets the bin statistics to zero.
   */
  inline void reset() noexcept {
    acc_wgt_.reset();
    acc_wgt_sq_.reset();
  }

  /*!
   * @brief Resets the bin to specific values.
   *
   * @param w The new sum of weights.
   * @param w2 The new sum of squared weights.
   */
  inline void reset(const T& w, const T& w2) noexcept {
    acc_wgt_.reset(w);
    acc_wgt_sq_.reset(w2);
  }

  /// @}

  /// @name Accessors
  /// @{

  /**
   * @brief Get the accumulated sum of weights.
   * @return The sum of weights ($W$).
   */
  [[nodiscard]] inline T weight() const noexcept {
    return acc_wgt_.result();
  }

  /**
   * @brief Get the accumulated sum of squared weights.
   * @return The sum of squared weights ($W^2$).
   */
  [[nodiscard]] inline T weight_sq() const noexcept {
    return acc_wgt_sq_.result();
  }

  /// @}

  /// @name Serialization
  /// @{

  /**
   * @brief Serializes the bin accumulator to an output stream.
   * @param out The output stream.
   */
  void serialize(std::ostream& out) const noexcept {
    acc_wgt_.serialize(out);
    acc_wgt_sq_.serialize(out);
  }

  /**
   * @brief Deserializes the bin accumulator from an input stream.
   * @param in The input stream.
   * @throws std::runtime_error If deserialization fails.
   */
  void deserialize(std::istream& in) {
    acc_wgt_.deserialize(in);
    acc_wgt_sq_.deserialize(in);
  }

  /// @}

};  // struct BinAccumulator

}  // namespace kakuhen::histogram
