#pragma once

#include "kakuhen/histogram/histogram_data.h"
#include "kakuhen/util/numeric_traits.h"
#include <cassert>
#include <span>
#include <vector>

namespace kakuhen::histogram {

/*!
 * @brief A view over a slice of the global histogram data.
 *
 * `HistogramView` acts as a handle to a specific histogram within the global
 * `HistogramData` storage. It manages the registration (allocation) of bins
 * and provides a convenient interface for filling multi-valued bins.
 *
 * It is designed to be lightweight and copyable.
 *
 * @tparam NT The numeric traits defining value type and index type.
 */
template <typename NT = util::num_traits_t<>>
class HistogramView {
 public:
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  // shorthands
  using S = size_type;
  using T = value_type;

  /*!
   * @brief Registers a new histogram view and allocates storage.
   *
   * @param data The global histogram data storage.
   * @param n_bins The number of bins in this histogram.
   * @param n_values_per_bin The number of values (observables) per bin.
   */
  HistogramView(HistogramData<NT>& data, S n_bins, S n_values_per_bin)
      : offset_(data.allocate(n_bins * n_values_per_bin)),
        n_bins_(n_bins),
        stride_(n_values_per_bin) {}

  /*!
   * @brief Fills the histogram buffer with a vector of values for a specific bin.
   *
   * This method maps the local bin index and the vector of values to the
   * corresponding global indices in the histogram buffer.
   *
   * @tparam Buffer The type of the histogram buffer (e.g., `HistogramBuffer`).
   * @tparam Range The type of the range of values (e.g., `std::vector`, `std::span`).
   * @param buffer The thread-local histogram buffer to fill.
   * @param local_bin_idx The local index of the bin (0 to n_bins - 1).
   * @param values The values to accumulate into the bin. Must have size == n_values_per_bin.
   */
  template <typename Buffer, typename Range>
  void fill(Buffer& buffer, S local_bin_idx, const Range& values) const {
    assert(local_bin_idx < n_bins_ && "Bin index out of bounds");
    assert(std::size(values) == stride_ && "Value count mismatch");

    const S base_global_idx = offset_ + local_bin_idx * stride_;
    
    // Using an index-based loop to support both std::vector and std::span
    // and to easily calculate the global index.
    S i = 0;
    for (const auto& val : values) {
      buffer.fill(base_global_idx + i, val);
      ++i;
    }
  }

  /*!
   * @brief Fills the histogram buffer with a single value (scalar).
   *
   * Optimized path for histograms with 1 value per bin.
   *
   * @param buffer The thread-local histogram buffer.
   * @param local_bin_idx The local index of the bin.
   * @param value The value to accumulate.
   */
  template <typename Buffer>
  void fill(Buffer& buffer, S local_bin_idx, const T& value) const {
    assert(local_bin_idx < n_bins_ && "Bin index out of bounds");
    assert(stride_ == 1 && "Value count mismatch (expected 1 for scalar fill)");
    buffer.fill(offset_ + local_bin_idx, value); // stride is 1, so idx is offset + local
  }

  /*!
   * @brief Get the global offset for this view.
   */
  [[nodiscard]] S offset() const noexcept { return offset_; }

  /*!
   * @brief Get the number of bins in this view.
   */
  [[nodiscard]] S n_bins() const noexcept { return n_bins_; }

  /*!
   * @brief Get the number of values per bin.
   */
  [[nodiscard]] S stride() const noexcept { return stride_; }

 private:
  S offset_;  //!< Starting index in the global storage.
  S n_bins_;  //!< Number of bins in this histogram.
  S stride_;  //!< Number of values per bin.
};

}  // namespace kakuhen::histogram
