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
  /*!
   * @brief Allocates a contiguous block of bins in the global storage.
   *
   * This method resizes the underlying storage to accommodate `n_bins`
   * additional bins.
   *
   * @param n_bins The number of additional bins to allocate.
   * @return The starting global index of the allocated block.
   * @throws std::length_error If the total number of bins exceeds the capacity of index type S.
   */
  S allocate(S n_bins) {
    const std::size_t current_size = bins_.size();
    const std::size_t count = static_cast<std::size_t>(n_bins);

    if (current_size + count > static_cast<std::size_t>(std::numeric_limits<S>::max())) {
      throw std::length_error("HistogramData: total bin count exceeds capacity of index type S.");
    }

    bins_.resize(current_size + count, bin_type{});
    return static_cast<S>(current_size);
  }

  /*!
   * @brief Accumulate a single weight into a bin.
   *
   * @param index The global bin index.
   * @param w The weight to accumulate.
   */
  void accumulate(S index, const T& w) {
    bins_[static_cast<std::size_t>(index)].accumulate(w);
  }

  /*!
   * @brief Accumulate weight and squared weight into a bin.
   *
   * @param index The global bin index.
   * @param w The sum of weights to add.
   * @param w2 The sum of squared weights to add.
   */
  void accumulate(S index, const T& w, const T& w2) {
    bins_[static_cast<std::size_t>(index)].accumulate(w, w2);
  }

  /*!
   * @brief Resets all bins and the event counter to zero.
   *
   * This preserves the number of bins and the capacity of the underlying
   * vector, avoiding reallocation.
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
   * This should be called once per event flush to track the total number
   * of samples contributed to the data.
   */
  void increment_count() noexcept {
    n_count_++;
  }

  /**
   * @brief Access the underlying vector of bins.
   * @return A reference to the vector of bin accumulators.
   */
  [[nodiscard]] std::vector<bin_type>& bins() noexcept {
    return bins_;
  }

  /**
   * @brief Access the underlying vector of bins (const).
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
   * @brief Checks if two HistogramData objects are identical.
   */
  [[nodiscard]] bool operator==(const HistogramData& other) const noexcept {
    return n_count_ == other.n_count_ && bins_ == other.bins_;
  }

  /**
   * @brief Checks if two HistogramData objects are different.
   */
  [[nodiscard]] bool operator!=(const HistogramData& other) const noexcept {
    return !(*this == other);
  }

  /// @}

  /// @name Serialization
  /// @{

  /**
   * @brief Serializes the histogram data to an output stream.
   *
   * @param out The output stream to write to.
   * @param with_type If true, prepends type identifiers for T, S, and U to the stream.
   */
  void serialize(std::ostream& out, bool with_type = false) const noexcept {
    if (with_type) {
      const int16_t T_tos = kakuhen::util::type::get_type_or_size<T>();
      const int16_t S_tos = kakuhen::util::type::get_type_or_size<S>();
      const int16_t U_tos = kakuhen::util::type::get_type_or_size<U>();
      kakuhen::util::serialize::serialize_one<int16_t>(out, T_tos);
      kakuhen::util::serialize::serialize_one<int16_t>(out, S_tos);
      kakuhen::util::serialize::serialize_one<int16_t>(out, U_tos);
    }
    kakuhen::util::serialize::serialize_one<U>(out, n_count_);
    kakuhen::util::serialize::serialize_one<S>(out, static_cast<S>(bins_.size()));
    kakuhen::util::serialize::serialize_container(out, bins_);
  }

  /**
   * @brief Deserializes the histogram data from an input stream.
   *
   * @param in The input stream to read from.
   * @param with_type If true, expects and verifies type identifiers for T, S, and U.
   * @throws std::runtime_error If type verification fails or the stream is corrupted.
   * @throws std::length_error If the bin count exceeds the capacity of the index type S.
   */
  void deserialize(std::istream& in, bool with_type = false) {
    if (with_type) {
      int16_t T_tos, S_tos, U_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, T_tos);
      if (T_tos != kakuhen::util::type::get_type_or_size<T>()) {
        throw std::runtime_error("HistogramData: type mismatch for value type T.");
      }
      kakuhen::util::serialize::deserialize_one<int16_t>(in, S_tos);
      if (S_tos != kakuhen::util::type::get_type_or_size<S>()) {
        throw std::runtime_error("HistogramData: type mismatch for index type S.");
      }
      kakuhen::util::serialize::deserialize_one<int16_t>(in, U_tos);
      if (U_tos != kakuhen::util::type::get_type_or_size<U>()) {
        throw std::runtime_error("HistogramData: type mismatch for count type U.");
      }
    }
    kakuhen::util::serialize::deserialize_one<U>(in, n_count_);
    S size_in;
    kakuhen::util::serialize::deserialize_one<S>(in, size_in);
    if (static_cast<std::uint64_t>(size_in) >
        static_cast<std::uint64_t>(std::numeric_limits<S>::max())) {
      throw std::length_error(
          "HistogramData: deserialized bin count exceeds capacity of index type S.");
    }
    bins_.resize(static_cast<std::size_t>(size_in));
    kakuhen::util::serialize::deserialize_container(in, bins_);
  }

  /// @}

  /// @name Management
  /// @{

  /**
   * @brief Reserves space for at least the specified number of bins.
   * @param capacity Total number of bins to reserve space for.
   */
  void reserve(S capacity) {
    bins_.reserve(static_cast<std::size_t>(capacity));
  }

  /**
   * @brief Clears all bins and resets the event counter.
   *
   * This removes all bin definitions from storage.
   */
  void clear() noexcept {
    bins_.clear();
    n_count_ = U(0);
  }

  /// @}

 private:
  std::vector<bin_type> bins_;  //!< Flattened storage for all histogram bins.
  U n_count_ = U(0);            //!< Total number of events contributed.
};

}  // namespace kakuhen::histogram
