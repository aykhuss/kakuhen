#pragma once

#include "kakuhen/histogram/histogram_data.h"
#include "kakuhen/util/numeric_traits.h"
#include <cassert>
#include <span>
#include <stdexcept>

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

  /**
   * @brief Constructs an empty HistogramView.
   */
  HistogramView() : offset_(0), n_bins_(0), stride_(0) {}

  /**
   * @brief Constructs a HistogramView from raw metadata.
   *
   * @param offset The starting global index.
   * @param n_bins The number of bins.
   * @param stride The number of values per bin.
   */
  HistogramView(S offset, S n_bins, S stride) : offset_(offset), n_bins_(n_bins), stride_(stride) {}

  /**
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

  /**
   * @brief Fills the histogram buffer with a span of values for a specific bin.
   *
   * This method maps the local bin index and the span of values to the
   * corresponding global indices in the histogram buffer.
   *
   * @tparam Buffer The type of the histogram buffer (e.g., `HistogramBuffer`).
   * @param buffer The thread-local histogram buffer to fill.
   * @param values The values to accumulate into the bin. Must have size == n_values_per_bin.
   * @param local_bin_idx The local index of the bin (0 to n_bins - 1).
   *
   * @note If `local_bin_idx` or the size of `values` is incorrect, this will assert in debug
   * builds.
   */
  template <typename Buffer>
  void fill_by_index(Buffer& buffer, std::span<const T> values, S local_bin_idx) const {
    assert(local_bin_idx < n_bins_ && "Bin index out of bounds");
    assert(values.size() == static_cast<std::size_t>(stride_) && "Value count mismatch");

    const S base_global_idx = offset_ + local_bin_idx * stride_;

    for (S i = 0; i < stride_; ++i) {
      buffer.fill(base_global_idx + i, values[i]);
    }
  }

  /**
   * @brief Fills the histogram buffer with a single value (scalar).
   *
   * Optimized path for histograms with 1 value per bin (stride == 1).
   *
   * @tparam Buffer The type of the histogram buffer.
   * @param buffer The thread-local histogram buffer.
   * @param value The value to accumulate.
   * @param local_bin_idx The local index of the bin.
   */
  template <typename Buffer>
  void fill_by_index(Buffer& buffer, const T& value, S local_bin_idx) const {
    assert(local_bin_idx < n_bins_ && "Bin index out of bounds");
    assert(stride_ == 1 && "Value count mismatch (expected 1 for scalar fill)");
    buffer.fill(offset_ + local_bin_idx, value);  // stride is 1, so idx is offset + local
  }

  /**
   * @brief Access the accumulator for a specific bin in this histogram view.
   *
   * @param data The global histogram data storage.
   * @param bin_idx The local bin index.
   * @param value_idx The value index within the bin (default 0).
   * @return A const reference to the bin accumulator.
   * @throws std::out_of_range If indices are out of bounds.
   */
  [[nodiscard]] const auto& get_bin(const HistogramData<NT>& data, S bin_idx,
                                    S value_idx = 0) const noexcept {
#ifndef NDEBUG
    if (bin_idx >= n_bins_) {
      assert(false && "HistogramView: bin index out of bounds");
    }
    if (value_idx >= stride_) {
      assert(false && "HistogramView: value index out of bounds");
    }
#endif
    const S global_idx = offset_ + bin_idx * stride_ + value_idx;
    return data.get_bin(global_idx);
  }

  /**
   * @brief Serializes the histogram view metadata.
   *
   * @param out The output stream to write to.
   * @param with_type If true, prepends type identifiers for T and S to the stream.
   */
  void serialize(std::ostream& out, bool with_type = false) const noexcept {
    if (with_type) {
      const int16_t T_tos = kakuhen::util::type::get_type_or_size<T>();
      const int16_t S_tos = kakuhen::util::type::get_type_or_size<S>();
      kakuhen::util::serialize::serialize_one<int16_t>(out, T_tos);
      kakuhen::util::serialize::serialize_one<int16_t>(out, S_tos);
    }
    kakuhen::util::serialize::serialize_one(out, offset_);
    kakuhen::util::serialize::serialize_one(out, n_bins_);
    kakuhen::util::serialize::serialize_one(out, stride_);
  }

  /**
   * @brief Deserializes the histogram view metadata.
   *
   * @param in The input stream to read from.
   * @param with_type If true, expects and verifies type identifiers for T and S.
   * @throws std::runtime_error If type verification fails or the stream is corrupted.
   */
  void deserialize(std::istream& in, bool with_type = false) {
    if (with_type) {
      int16_t T_tos, S_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, T_tos);
      if (T_tos != kakuhen::util::type::get_type_or_size<T>()) {
        throw std::runtime_error("HistogramView: type mismatch for value type T.");
      }
      kakuhen::util::serialize::deserialize_one<int16_t>(in, S_tos);
      if (S_tos != kakuhen::util::type::get_type_or_size<S>()) {
        throw std::runtime_error("HistogramView: type mismatch for index type S.");
      }
    }
    kakuhen::util::serialize::deserialize_one(in, offset_);
    kakuhen::util::serialize::deserialize_one(in, n_bins_);
    kakuhen::util::serialize::deserialize_one(in, stride_);
  }

  /**
   * @brief Get the global offset for this view in `HistogramData`.
   * @return Starting index.
   */
  [[nodiscard]] S offset() const noexcept {
    return offset_;
  }

  /**
   * @brief Get the number of bins in this view.
   * @return Bin count.
   */
  [[nodiscard]] S n_bins() const noexcept {
    return n_bins_;
  }

  /**
   * @brief Get the number of values per bin (stride).
   * @return Values per bin.
   */
  [[nodiscard]] S stride() const noexcept {
    return stride_;
  }

 private:
  S offset_;  //!< Starting index in the global storage.
  S n_bins_;  //!< Number of bins in this histogram.
  S stride_;  //!< Number of values per bin.
};

}  // namespace kakuhen::histogram
