#pragma once

#include "kakuhen/histogram/bin_accumulator.h"
#include "kakuhen/util/numeric_traits.h"
#include "kakuhen/util/serialize.h"
#include <vector>

namespace kakuhen::histogram {

/*!
 * @brief Global storage for histogram data.
 *
 * This class acts as the central repository for all histogram bins in the registry.
 * It stores a flattened vector of `BinAccumulator` objects, which are updated
 * by flushing (thread-local) `HistogramBuffer`s.
 *
 * It ensures that the final aggregated results maintain high precision (via
 * `BinAccumulator`'s default TwoSum strategy) while allowing efficient
 * lock-free accumulation during the event filling phase via buffers.
 *
 * It also tracks the total number of events (flushes) contributed to the data,
 * which is essential for proper normalization of the results (e.g. cross-section
 * calculation).
 *
 * @tparam NT The numeric traits defining value type and index type.
 */
template <typename NT = util::num_traits_t<>>
class HistogramData {
 public:
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  using count_type = typename num_traits::count_type;
  using bin_type = BinAccumulator<value_type>;
  // shorthands
  using S = size_type;
  using T = value_type;
  using U = count_type;

  /*!
   * @brief Allocates a contiguous block of bins in the global storage.
   *
   * This method allows for incremental registration of histograms. It resizes
   * the underlying storage to accommodate `n_bins` additional bins.
   *
   * @param n_bins The number of bins to allocate.
   * @return The starting global index of the allocated block.
   */
  S allocate(S n_bins) {
    S start_index = size();
    bins_.resize(start_index + n_bins, bin_type{});
    return start_index;
  }

  /*!
   * @brief Accumulate a single weight into a bin.
   *
   * @param index The global bin index.
   * @param w The weight to accumulate.
   */
  void accumulate(S index, const T& w) {
    bins_[index].accumulate(w);
  }

  /*!
   * @brief Accumulate weight and squared weight into a bin.
   *
   * @param index The global bin index.
   * @param w The sum of weights to add.
   * @param w2 The sum of squared weights to add.
   */
  void accumulate(S index, const T& w, const T& w2) {
    bins_[index].accumulate(w, w2);
  }

  /*!
   * @brief Resets all bins and the event counter to zero.
   *
   * This preserves the number of bins and the capacity of the vector,
   * avoiding reallocation.
   */
  void reset() noexcept {
    for (auto& bin : bins_) {
      bin.reset();
    }
    n_count_ = U(0);
  }

  /*!
   * @brief Increment the total event counter.
   *
   * This should be called once per event flush.
   */
  void increment_count() noexcept {
    n_count_++;
  }

  /**
   * @brief Access the underlying vector of bins.
   *
   * This is used by `HistogramBuffer::flush` to write aggregated data.
   *
   * @return A reference to the vector of bin accumulators.
   */
  [[nodiscard]] std::vector<bin_type>& bins() noexcept {
    return bins_;
  }

  /**
   * @brief Access the underlying vector of bins (const).
   *
   * @return A const reference to the vector of bin accumulators.
   */
  [[nodiscard]] const std::vector<bin_type>& bins() const noexcept {
    return bins_;
  }

  /**
   * @brief Get the number of bins in storage.
   * @return Total bin count.
   */
  [[nodiscard]] inline S size() const noexcept {
    return static_cast<S>(bins_.size());
  }

  /**
   * @brief Get the total number of accumulated events (flushes).
   * @return Total event count.
   */
  [[nodiscard]] inline U count() const noexcept {
    return n_count_;
  }

  /**
   * @brief Serializes the histogram data to an output stream.
   * @param out The output stream.
   */
  void serialize(std::ostream& out) const noexcept {
    kakuhen::util::serialize::serialize_one<U>(out, n_count_);
    kakuhen::util::serialize::serialize_one<S>(out, static_cast<S>(bins_.size()));
    kakuhen::util::serialize::serialize_container(out, bins_);
  }

  /**
   * @brief Deserializes the histogram data from an input stream.
   * @param in The input stream.
   * @throws std::runtime_error If deserialization fails.
   */
  void deserialize(std::istream& in) {
    kakuhen::util::serialize::deserialize_one<U>(in, n_count_);
    S size_in;
    kakuhen::util::serialize::deserialize_one<S>(in, size_in);
    bins_.resize(static_cast<std::size_t>(size_in));
    kakuhen::util::serialize::deserialize_container(in, bins_);
  }

 private:
  std::vector<bin_type> bins_;  //!< Flattened storage for all histogram bins.
  U n_count_ = U(0);            //!< Total number of events contributed.
};

}  // namespace kakuhen::histogram
