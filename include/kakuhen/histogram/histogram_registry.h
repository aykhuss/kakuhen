#pragma once

#include "kakuhen/histogram/histogram_buffer.h"
#include "kakuhen/histogram/histogram_data.h"
#include "kakuhen/histogram/histogram_id.h"
#include "kakuhen/histogram/histogram_view.h"
#include "kakuhen/util/numeric_traits.h"
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>

namespace kakuhen::histogram {

/*!
 * @brief Manages the lifecycle and registration of multiple histograms.
 *
 * The `HistogramRegistry` serves as a facade over `HistogramData`, simplifying
 * the process of defining histograms ("booking") and creating compatible
 * thread-local buffers.
 *
 * It stores the `HistogramData` and maintains a list of registered `HistogramView`s
 * associated with human-readable names, accessible via `HistogramId`.
 *
 * @tparam NT The numeric traits defining value type and index type.
 */
template <typename NT = util::num_traits_t<>>
class HistogramRegistry {
 public:
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  // shorthands
  using S = size_type;
  using T = value_type;
  using View = HistogramView<NT>;
  using Id = HistogramId<S>;

  /*!
   * @brief Books a new histogram.
   *
   * Allocates space in the underlying data storage and returns a unique ID
   * for the histogram.
   *
   * @param name A unique identifier for the histogram.
   * @param n_bins The number of bins.
   * @param n_values_per_bin The number of values per bin (default 1).
   * @return A `HistogramId` handle to the booked histogram.
   */
  [[nodiscard]] Id book(std::string_view name, S n_bins, S n_values_per_bin = 1) {
    names_.emplace_back(name);
    views_.emplace_back(data_, n_bins, n_values_per_bin);
    return Id{static_cast<S>(views_.size() - 1)};
  }

  /*!
   * @brief Fills a registered histogram with a range of values.
   *
   * @param buffer The thread-local buffer.
   * @param id The histogram ID.
   * @param local_bin_idx The local bin index.
   * @param values The values to accumulate.
   */
  template <typename Buffer, typename Range>
  void fill(Buffer& buffer, Id id, S local_bin_idx, const Range& values) const {
    views_[id.id()].fill(buffer, local_bin_idx, values);
  }

  /*!
   * @brief Fills a registered histogram with a single value.
   *
   * @param buffer The thread-local buffer.
   * @param id The histogram ID.
   * @param local_bin_idx The local bin index.
   * @param value The value to accumulate.
   */
  template <typename Buffer>
  void fill(Buffer& buffer, Id id, S local_bin_idx, const T& value) const {
    views_[id.id()].fill(buffer, local_bin_idx, value);
  }

  /*!
   * @brief Flushes a buffer into the registry's data.
   *
   * @param buffer The thread-local buffer to flush.
   */
  template <typename Buffer>
  void flush(Buffer& buffer) {
    buffer.flush(data_);
  }

  /*!
   * @brief Creates and initializes a thread-local buffer compatible with this registry.
   *
   * This helper ensures the buffer is initialized with the correct total size
   * matching the currently booked histograms.
   *
   * @tparam Acc The accumulator strategy for the buffer (default: TwoSum via buffer default).
   * @param reserve_size Initial capacity for the local dense buffers (default: 1024).
   * @return A ready-to-use `HistogramBuffer`.
   */
  template <typename Acc = kakuhen::util::accumulator::Accumulator<T>>
  auto create_buffer(S reserve_size = 1024) const {
    HistogramBuffer<NT, Acc> buffer;
    buffer.init(data_.size(), reserve_size);
    return buffer;
  }

  /*!
   * @brief Access the underlying global data storage.
   */
  HistogramData<NT>& data() noexcept { return data_; }

  /*!
   * @brief Access the underlying global data storage (const).
   */
  const HistogramData<NT>& data() const noexcept { return data_; }

  /*!
   * @brief Retrieve the list of all registered histogram IDs.
   */
  std::vector<Id> ids() const {
    std::vector<Id> result;
    result.reserve(views_.size());
    for (S i = 0; i < static_cast<S>(views_.size()); ++i) {
      result.emplace_back(i);
    }
    return result;
  }

  /*!
   * @brief Retrieve the view associated with an ID.
   * @param id The histogram ID.
   */
  View get_view(Id id) const {
    // We assume ID is valid as it comes from a typed wrapper.
    // Could add range check here.
    return views_[id.id()];
  }

  /*!
   * @brief Retrieve the name associated with an ID.
   * @param id The histogram ID.
   */
  std::string_view get_name(Id id) const {
    return names_[id.id()];
  }

  /*!
   * @brief Look up a histogram ID by name.
   * 
   * @note This is O(N).
   * @throws std::runtime_error if name not found.
   */
  Id get_id(std::string_view name) const {
      for (size_t i = 0; i < names_.size(); ++i) {
          if (names_[i] == name) return Id{static_cast<S>(i)};
      }
      throw std::runtime_error("Histogram not found: " + std::string(name));
  }

 private:
  HistogramData<NT> data_;
  std::vector<std::string> names_;
  std::vector<View> views_;
};

}  // namespace kakuhen::histogram
