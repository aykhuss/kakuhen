#pragma once

#include "kakuhen/integrator/integral_accumulator.h"
#include "kakuhen/util/serialize.h"
#include "kakuhen/util/type.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace kakuhen::integrator {

/*!
 * @brief Stores and processes results from multiple integral accumulations.
 *
 * This class is designed to accumulate and analyze results from multiple
 * independent integral calculations. It can compute the weighted mean,
 * variance, error, chi-squared, and degrees of freedom from a collection
 * of `IntegralAccumulator` objects.
 *
 * @tparam T The value type for the integral results (e.g., double).
 * @tparam U The count type for the number of evaluations (e.g., uint64_t).
 */
template <typename T, typename U>
class Result {
 public:
  using value_type = T;
  using count_type = U;
  using int_acc_type = IntegralAccumulator<T, U>;

  /*!
   * @brief Accumulate a single integral result.
   *
   * Adds an `IntegralAccumulator` object to the collection of results.
   *
   * @param acc The `IntegralAccumulator` object to add.
   */
  void accumulate(const int_acc_type& acc) {
    if (acc.count() == U(0)) return;  // skip empty
    results_.push_back(acc);
  }

  /*!
   * @brief Accumulate results from another Result object.
   *
   * Appends all `IntegralAccumulator` objects from another `Result` instance
   * to this collection.
   *
   * @param res The `Result` object whose contents will be accumulated.
   */
  void accumulate(const Result<T, U>& res) {
    for (const auto& acc : res.results_) {
      if (acc.count() == U(0)) continue;
      results_.push_back(acc);
    }
  }

  /*!
   * @brief Clears all accumulated results.
   */
  void reset() {
    results_.clear();
  }

  /// @name Queries
  /// @{

  /*!
   * @brief Get the number of individual integral results accumulated.
   *
   * @return The number of accumulated `IntegralAccumulator` objects.
   */
  [[nodiscard]] U size() const noexcept {
    return static_cast<U>(results_.size());
  }

  /*!
   * @brief Get the total number of function evaluations across all accumulated results.
   *
   * @return The sum of `count()` from all `IntegralAccumulator` objects.
   */
  [[nodiscard]] inline U count() const noexcept {
    U n_tot = U(0);
    for (const auto& r : results_) {
      n_tot += r.count();
    }
    return n_tot;
  }

  /*!
   * @brief Computes the weighted mean of the accumulated integral results.
   *
   * The weighting is done by the inverse variance of each individual result.
   *
   * @return The weighted mean value of the integral.
   * @throws std::runtime_error if no results have been accumulated.
   */
  [[nodiscard]] T value() const {
    if (results_.empty()) throw std::runtime_error("No results to average");

    T sum_wgt = T(0);
    T sum_val_wgt = T(0);

    for (const auto& r : results_) {
      T variance = r.variance();
      if (variance <= T(0)) continue;  // skip invalid/zero variance
      T wgt = T(1) / variance;
      sum_wgt += wgt;
      sum_val_wgt += r.value() * wgt;
    }

    if (sum_wgt == T(0)) {
      // Fallback if all variances are zero (e.g. deterministic): use arithmetic mean
      // or just return the first value? Arithmetic mean is safer.
      T sum_val = T(0);
      for (const auto& r : results_)
        sum_val += r.value();
      return sum_val / static_cast<T>(results_.size());
    }

    return sum_val_wgt / sum_wgt;
  }

  /*!
   * @brief Computes the variance of the weighted mean.
   *
   * @return The variance of the weighted mean.
   * @throws std::runtime_error if no results have been accumulated.
   */
  [[nodiscard]] T variance() const {
    if (results_.empty()) throw std::runtime_error("No results to average");

    T sum_wgt = T(0);
    for (const auto& r : results_) {
      T variance = r.variance();
      if (variance <= T(0)) continue;  // skip invalid
      sum_wgt += T(1) / variance;
    }

    if (sum_wgt == T(0)) return T(0);  // All variances were zero -> zero error

    return T(1) / sum_wgt;
  }

  /*!
   * @brief Computes the error (standard deviation) of the weighted mean.
   *
   * This is the square root of the variance.
   *
   * @return The error of the weighted mean.
   */
  [[nodiscard]] T error() const {
    return std::sqrt(variance());
  }

  /*!
   * @brief Computes the chi-squared value of the accumulated results.
   *
   * This measures the consistency of the individual results with their
   * weighted mean.
   *
   * @return The chi-squared value. Returns 0 if there are less than 2 results.
   */
  [[nodiscard]] T chi2() const {
    if (results_.size() < 2) return T(0);

    T mean = value();
    T chi2 = T(0);

    for (const auto& r : results_) {
      T variance = r.variance();
      if (variance <= T(0)) continue;  // skip invalid
      T diff = r.value() - mean;
      chi2 += (diff * diff) / variance;
    }
    return chi2;
  }

  /*!
   * @brief Get the degrees of freedom for the chi-squared calculation.
   *
   * @return The number of results minus one, or zero if there's one or fewer results.
   */
  [[nodiscard]] U dof() const {
    return results_.size() > 1 ? static_cast<U>(results_.size() - 1) : U(0);
  }

  /*!
   * @brief Computes the chi-squared per degree of freedom (chi2/dof).
   *
   * This is a common metric for assessing the goodness-of-fit or consistency.
   *
   * @return The chi-squared per degree of freedom. Returns 0 if there are
   * less than 2 results.
   */
  [[nodiscard]] T chi2dof() const {
    if (results_.size() < 2) return T(0);
    return chi2() / static_cast<T>(dof());
  }

  /// @}

  /// @name Serialization
  /// @{

  /*!
   * @brief Serializes the result object to an output stream.
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
    kakuhen::util::serialize::serialize_size(out, results_.size());
    kakuhen::util::serialize::serialize_container(out, results_);
  }

  /*!
   * @brief Deserializes the result object from an input stream.
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
    std::size_t size_in = 0;
    kakuhen::util::serialize::deserialize_size(in, size_in);
    results_.resize(size_in);
    kakuhen::util::serialize::deserialize_container(in, results_);
  }
  /// @}

 private:
  std::vector<int_acc_type> results_;

};  // class Result

}  // namespace kakuhen::integrator
