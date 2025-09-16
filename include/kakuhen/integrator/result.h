#pragma once

#include "kakuhen/integrator/integral_accumulator.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace kakuhen::integrator {

template <typename T, typename U>
class Result {
public:
  using value_type = T;
  using count_type = U;
  using int_acc_type = IntegralAccumulator<T, U>;

  // store results of independent integrations
  void accumulate(const int_acc_type& acc) {
    if (acc.count() == U(0)) return;  // skip empty
    results_.push_back(acc);
  }

  void reset() {
    results_.clear();
  }

  U size() const noexcept {
    return results_.size();
  }

  // total number of calls
  inline U count() const noexcept {
    U n_tot = U(0);
    for (const auto& r : results_) {
      n_tot += r.count();
    }
    return n_tot;
  }

  // weighted mean of results (by inverse variance)
  T value() const {
    if (results_.empty()) throw std::runtime_error("No results to average");

    T sum_wgt = T(0);
    T sum_val_wgt = T(0);

    for (const auto& r : results_) {
      T variance = r.variance();
      if (variance <= T(0)) continue;  // skip invalid
      T wgt = T(1) / variance;
      sum_wgt += wgt;
      sum_val_wgt += r.value() * wgt;
    }
    return sum_val_wgt / sum_wgt;
  }

  // variance of the weighted mean
  T variance() const {
    if (results_.empty()) throw std::runtime_error("No results to average");

    T sum_wgt = T(0);
    for (const auto& r : results_) {
      T variance = r.variance();
      if (variance <= T(0)) continue;  // skip invalid
      sum_wgt += T(1) / variance;
    }
    return T(1) / sum_wgt;
  }

  T error() const {
    return std::sqrt(variance());
  }

  // chi^2 of results against weighted mean
  T chi2() const {
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

  // degrees of freedom
  U dof() const {
    return results_.size() > 1 ? results_.size() - 1 : 0;
  }

  // chi^2/dof of results
  T chi2dof() const {
    if (results_.size() < 2) return 0;
    return chi2() / T(dof());
  }

 private:
  std::vector<int_acc_type> results_;

};  // class Result

}  // namespace kakuhen::integrator
