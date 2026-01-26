#pragma once

#include "kakuhen/histogram/axis_data.h"
#include "kakuhen/histogram/axis_id.h"
#include "kakuhen/histogram/axis_view.h"
#include "kakuhen/histogram/histogram_buffer.h"
#include "kakuhen/histogram/histogram_data.h"
#include "kakuhen/histogram/histogram_id.h"
#include "kakuhen/histogram/histogram_view.h"
#include "kakuhen/util/math.h"
#include "kakuhen/util/numeric_traits.h"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace kakuhen::histogram {

/**
 * @brief Manages the lifecycle and registration of multiple histograms.
 *
 * The `HistogramRegistry` serves as a facade over `HistogramData` and `AxisData`.
 * It manages:
 * 1. Global bin storage (HistogramData)
 * 2. Global axis definition storage (AxisData)
 * 3. Registered Axes (list of AxisViews)
 * 4. Registered Histograms (mapping of Name -> HistogramView + AxisId)
 *
 * This class provides a centralized point for creating, booking, and filling
 * histograms, ensuring that memory allocation and axis mapping are handled
 * consistently.
 *
 * @tparam NT The numeric traits defining coordinate, index, and count types.
 */
template <typename NT = util::num_traits_t<>>
class HistogramRegistry {
 public:
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  using count_type = typename num_traits::count_type;
  // shorthands
  using S = size_type;
  using T = value_type;
  using U = count_type;
  using View = HistogramView<NT>;
  using Id = HistogramId<S>;
  using AxId = AxisId<S>;
  using AxisVar = AxisVariant<T, S>;

  /**
   * @brief Creates and registers a new axis.
   *
   * @tparam AxisType The type of axis (UniformAxis or VariableAxis).
   * @tparam Args Constructor argument types.
   * @param args Constructor arguments for the axis (excluding AxisData).
   * @return The ID of the registered axis.
   */
  template <typename AxisType, typename... Args>
  [[nodiscard]] AxId create_axis(Args&&... args)
    requires(std::is_same_v<AxisType, UniformAxis<T, S>> ||
             std::is_same_v<AxisType, VariableAxis<T, S>>)
  {
    axes_.emplace_back(std::in_place_type<AxisType>, axis_data_, std::forward<Args>(args)...);
    return AxId{static_cast<S>(axes_.size() - 1)};
  }

  /**
   * @brief Creates and registers a new axis using an initializer list.
   *
   * @tparam AxisType The type of axis (must be VariableAxis).
   * @param list The initializer list of bin edges.
   * @return The ID of the registered axis.
   */
  template <typename AxisType>
  [[nodiscard]] AxId create_axis(std::initializer_list<T> list)
    requires(std::is_same_v<AxisType, VariableAxis<T, S>>)
  {
    return create_axis<AxisType>(std::vector<T>(list));
  }

  /**
   * @brief Books a new histogram using an existing axis ID.
   *
   * @param name A unique identifier for the histogram.
   * @param axis_id The ID of the registered axis.
   * @param n_values_per_bin The number of values per bin.
   * @return A `HistogramId` handle.
   * @throws std::out_of_range If the axis_id is invalid.
   * @throws std::invalid_argument If the name is already in use or axis is None.
   */
  [[nodiscard]] Id book(std::string_view name, AxId axis_id, S n_values_per_bin = 1) {
    if (axis_id.id() >= axes_.size()) {
      throw std::out_of_range("HistogramRegistry: invalid AxisId.");
    }

    const auto& axis_var = axes_[axis_id.id()];

    if (std::holds_alternative<std::monostate>(axis_var)) {
      throw std::invalid_argument(
          "HistogramRegistry: cannot book with None axis using this method.");
    }

    S n_bins = std::visit(
        [](const auto& ax) -> S {
          using Type = std::decay_t<decltype(ax)>;
          if constexpr (std::is_same_v<Type, std::monostate>) {
            return 0;  // Should not happen due to check above
          } else {
            return ax.n_bins();
          }
        },
        axis_var);

    return book_with_id(name, axis_id, n_bins, n_values_per_bin);
  }

  /**
   * @brief Books a new histogram without an associated axis (view only).
   *
   * @param name A unique identifier for the histogram.
   * @param n_bins The number of bins to allocate.
   * @param n_values_per_bin The number of values per bin (default 1).
   * @return A `HistogramId` handle.
   */
  [[nodiscard]] Id book(std::string_view name, S n_bins, S n_values_per_bin = 1) {
    axes_.emplace_back(std::monostate{});
    AxId ax_id{static_cast<S>(axes_.size() - 1)};
    return book_with_id(name, ax_id, n_bins, n_values_per_bin);
  }

  /**
   * @brief Fills a registered histogram with a span of values using a local bin index.
   *
   * @tparam Buffer The type of the histogram buffer.
   * @param buffer The thread-local buffer to fill.
   * @param id The ID of the histogram.
   * @param local_bin_idx The index of the bin within the histogram.
   * @param values The span of values to accumulate.
   */
  template <typename Buffer>
  void fill(Buffer& buffer, Id id, S local_bin_idx, std::span<const T> values) const {
    assert(id.id() < entries_.size());
    entries_[id.id()].view.fill(buffer, local_bin_idx, values);
  }

  /**
   * @brief Fills a registered histogram with a single value using a local bin index.
   *
   * @tparam Buffer The type of the histogram buffer.
   * @param buffer The thread-local buffer to fill.
   * @param id The ID of the histogram.
   * @param local_bin_idx The index of the bin within the histogram.
   * @param value The value to accumulate.
   */
  template <typename Buffer>
  void fill(Buffer& buffer, Id id, S local_bin_idx, const T& value) const {
    assert(id.id() < entries_.size());
    entries_[id.id()].view.fill(buffer, local_bin_idx, value);
  }

  /**
   * @brief Fills a registered histogram with a span of values by mapping an x-coordinate.
   *
   * @tparam Buffer The type of the histogram buffer.
   * @param buffer The thread-local buffer to fill.
   * @param id The ID of the histogram.
   * @param x The x-coordinate to map to a bin.
   * @param values The span of values to accumulate.
   */
  template <typename Buffer>
  void fill(Buffer& buffer, Id id, const T& x, std::span<const T> values) const {
    assert(id.id() < entries_.size());
    const auto& entry = entries_[id.id()];
    const S bin_idx = get_axis_index(entry.axis_id, x);
    entry.view.fill(buffer, bin_idx, values);
  }

  /**
   * @brief Fills a registered histogram with a single value by mapping an x-coordinate.
   *
   * @tparam Buffer The type of the histogram buffer.
   * @param buffer The thread-local buffer to fill.
   * @param id The ID of the histogram.
   * @param x The x-coordinate to map to a bin.
   * @param value The value to accumulate.
   */
  template <typename Buffer>
  void fill(Buffer& buffer, Id id, const T& x, const T& value) const {
    assert(id.id() < entries_.size());
    const auto& entry = entries_[id.id()];
    const S bin_idx = get_axis_index(entry.axis_id, x);
    entry.view.fill(buffer, bin_idx, value);
  }

  /**
   * @brief Flushes a buffer into the registry's global data storage.
   *
   * This transfers accumulated weights from a thread-local buffer to the
   * shared global storage.
   *
   * @tparam Buffer The type of the histogram buffer.
   * @param buffer The buffer to flush.
   */
  template <typename Buffer>
  void flush(Buffer& buffer) {
    buffer.flush(data_);
  }

  /**
   * @brief Creates and initializes a thread-local buffer for filling histograms.
   *
   * @tparam Acc The accumulator type to use (e.g., TwoSumAccumulator).
   * @return A newly initialized `HistogramBuffer`.
   */
  template <typename Acc = kakuhen::util::accumulator::Accumulator<T>>
  [[nodiscard]] auto create_buffer() const {
    HistogramBuffer<NT, Acc> buffer;
    const S reserve_size = num_entries();
    buffer.init(data_.size(), reserve_size);
    return buffer;
  }

  /// @name Accessors
  /// @{

  /**
   * @brief Access the underlying global bin storage.
   * @return A reference to the global `HistogramData`.
   */
  [[nodiscard]] HistogramData<NT>& data() noexcept {
    return data_;
  }

  /**
   * @brief Access the underlying global bin storage (const).
   * @return A const reference to the global `HistogramData`.
   */
  [[nodiscard]] const HistogramData<NT>& data() const noexcept {
    return data_;
  }

  /**
   * @brief Access the underlying axis parameter storage.
   * @return A reference to the global `AxisData`.
   */
  [[nodiscard]] AxisData<T, S>& axis_data() noexcept {
    return axis_data_;
  }

  /**
   * @brief Access the underlying axis parameter storage (const).
   * @return A const reference to the global `AxisData`.
   */
  [[nodiscard]] const AxisData<T, S>& axis_data() const noexcept {
    return axis_data_;
  }

  /**
   * @brief Access the accumulator for a specific bin in a registered histogram.
   *
   * @param id The histogram ID.
   * @param bin_idx The local bin index.
   * @param value_idx The value index within the bin (default 0).
   * @return A const reference to the bin accumulator.
   * @throws std::out_of_range If the ID or indices are invalid.
   */
  [[nodiscard]] const auto& get_bin(Id id, S bin_idx, S value_idx = 0) const {
    if (id.id() >= static_cast<S>(entries_.size())) {
      throw std::out_of_range("HistogramRegistry: invalid HistogramId.");
    }
    return entries_[id.id()].view.get_bin(data_, bin_idx, value_idx);
  }

  /**
   * @brief Get the mean value (sum of weights / N) for a bin.
   *
   * @param id The histogram ID.
   * @param bin_idx The local bin index.
   * @param value_idx The value index within the bin (default 0).
   * @return The mean value.
   */
  [[nodiscard]] T value(Id id, S bin_idx, S value_idx = 0) const {
    const auto& bin = get_bin(id, bin_idx, value_idx);
    const T n = static_cast<T>(data_.count());
    if (n == T(0)) return T(0);
    return bin.weight() / n;
  }

  /**
   * @brief Get the variance of the mean value for a bin.
   *
   * @param id The histogram ID.
   * @param bin_idx The local bin index.
   * @param value_idx The value index within the bin (default 0).
   * @return The variance of the mean.
   */
  [[nodiscard]] T variance(Id id, S bin_idx, S value_idx = 0) const {
    const U n_count = data_.count();
    if (n_count <= 1) return T(0);

    const auto& bin = get_bin(id, bin_idx, value_idx);
    const T n = static_cast<T>(n_count);
    const T mean = bin.weight() / n;
    return (bin.weight_sq() / n - mean * mean) / (n - T(1));
  }

  /**
   * @brief Get the statistical error (standard deviation of the mean) for a bin.
   *
   * @param id The histogram ID.
   * @param bin_idx The local bin index.
   * @param value_idx The value index within the bin (default 0).
   * @return The statistical error.
   */
  [[nodiscard]] T error(Id id, S bin_idx, S value_idx = 0) const {
    return std::sqrt(variance(id, bin_idx, value_idx));
  }

  /**
   * @brief Retrieve the list of all registered histogram IDs.
   * @return A vector of `HistogramId` handles.
   */
  [[nodiscard]] std::vector<Id> ids() const {
    std::vector<Id> result;
    result.reserve(entries_.size());
    for (S i = 0; i < static_cast<S>(entries_.size()); ++i) {
      result.emplace_back(i);
    }
    return result;
  }

  /**
   * @brief Retrieve the view handle for a specific histogram.
   *
   * @param id The ID of the histogram.
   * @return The `HistogramView` associated with the ID.
   */
  [[nodiscard]] View get_view(Id id) const {
    assert(id.id() < entries_.size());
    return entries_[id.id()].view;
  }

  /**
   * @brief Retrieve the human-readable name for a specific histogram.
   *
   * @param id The ID of the histogram.
   * @return The name registered for this histogram.
   */
  [[nodiscard]] std::string_view get_name(Id id) const {
    assert(id.id() < names_.size());
    return names_[id.id()];
  }

  /**
   * @brief Look up a histogram's unique ID by its registered name.
   *
   * @param name The name to look up.
   * @return The unique `HistogramId`.
   * @throws std::runtime_error If the name is not found.
   * @note This lookup is O(N) where N is the number of registered histograms.
   */
  [[nodiscard]] Id get_id(std::string_view name) const {
    for (size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) return Id{static_cast<S>(i)};
    }
    throw std::runtime_error("HistogramRegistry: not found: " + std::string(name));
  }

  /**
   * @brief Returns the total number of registered entries.
   * @return The number of histograms.
   */
  [[nodiscard]] S num_entries() const noexcept {
    return static_cast<S>(entries_.size());
  }

  /// @}

  /// @name Serialization
  /// @{

  /**
   * @brief Serializes the entire registry state to an output stream.
   *
   * @param out The output stream to write to.
   * @param with_type If true, verifies traits compatibility during deserialization.
   */
  void serialize(std::ostream& out, bool with_type = true) const noexcept {
    if (with_type) {
      const int16_t T_tos = kakuhen::util::type::get_type_or_size<T>();
      const int16_t S_tos = kakuhen::util::type::get_type_or_size<S>();
      const int16_t U_tos = kakuhen::util::type::get_type_or_size<U>();
      kakuhen::util::serialize::serialize_one<int16_t>(out, T_tos);
      kakuhen::util::serialize::serialize_one<int16_t>(out, S_tos);
      kakuhen::util::serialize::serialize_one<int16_t>(out, U_tos);
    }

    // 1. Serialize Global Storage
    data_.serialize(out);
    axis_data_.serialize(out);

    // 2. Serialize Metadata
    util::serialize::serialize_size(out, names_.size());
    for (const auto& name : names_) {
      util::serialize::serialize_one(out, name);
    }

    // 3. Serialize Axes (using metadata-integrated tagging)
    util::serialize::serialize_size(out, axes_.size());
    for (const auto& var : axes_) {
      std::visit(
          [&](const auto& ax) {
            using Type = std::decay_t<decltype(ax)>;
            if constexpr (std::is_same_v<Type, std::monostate>) {
              AxisMetadata<T, S>{AxisType::None}.serialize(out);
            } else {
              ax.serialize(out);
            }
          },
          var);
    }

    // 4. Serialize Entries
    util::serialize::serialize_size(out, entries_.size());
    for (const auto& entry : entries_) {
      entry.serialize(out);
    }
  }

  /**
   * @brief Deserializes the entire registry state from an input stream.
   *
   * @param in The input stream to read from.
   * @param with_type If true, verifies traits compatibility.
   * @throws std::runtime_error If traits mismatch or deserialization fails.
   */
  void deserialize(std::istream& in, bool with_type = true) {
    if (with_type) {
      int16_t T_tos, S_tos, U_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, T_tos);
      if (T_tos != kakuhen::util::type::get_type_or_size<T>()) {
        throw std::runtime_error("HistogramRegistry: type mismatch for value type T.");
      }
      kakuhen::util::serialize::deserialize_one<int16_t>(in, S_tos);
      if (S_tos != kakuhen::util::type::get_type_or_size<S>()) {
        throw std::runtime_error("HistogramRegistry: type mismatch for index type S.");
      }
      kakuhen::util::serialize::deserialize_one<int16_t>(in, U_tos);
      if (U_tos != kakuhen::util::type::get_type_or_size<U>()) {
        throw std::runtime_error("HistogramRegistry: type mismatch for count type U.");
      }
    }

    // 1. Deserialize Global Storage
    data_.deserialize(in);
    axis_data_.deserialize(in);

    // 2. Deserialize Metadata
    std::size_t n_names;
    util::serialize::deserialize_size(in, n_names);
    names_.resize(n_names);
    for (auto& name : names_) {
      util::serialize::deserialize_one(in, name);
    }

    // 3. Deserialize Axes
    std::size_t n_axes;
    util::serialize::deserialize_size(in, n_axes);
    axes_.clear();
    axes_.reserve(n_axes);
    for (std::size_t i = 0; i < n_axes; ++i) {
      AxisMetadata<T, S> meta;
      meta.deserialize(in);
      axes_.emplace_back(restore_axis(meta));
    }

    // 4. Deserialize Entries
    std::size_t n_entries;
    util::serialize::deserialize_size(in, n_entries);
    entries_.clear();
    entries_.reserve(n_entries);
    for (std::size_t i = 0; i < n_entries; ++i) {
      Entry entry;
      entry.deserialize(in);
      entries_.push_back(std::move(entry));
    }
  }

  /// @}

  /// @name Management
  /// @{

  /**
   * @brief Clears all registered histograms and axis definitions.
   */
  void clear() noexcept {
    data_.clear();
    axis_data_.clear();
    entries_.clear();
    axes_.clear();
    names_.clear();
  }

  /// @}

 private:
  /**
   * @brief Internal structure mapping a histogram handle to its components.
   */
  struct Entry {
    AxId axis_id{0};     //!< ID of the axis definition used.
    View view{0, 0, 0};  //!< Handle to the physical bin storage.

    /**
     * @brief Serialize the entry metadata.
     * @param out The output stream.
     */
    void serialize(std::ostream& out) const noexcept {
      util::serialize::serialize_one(out, axis_id.id());
      view.serialize(out);
    }

    /**
     * @brief Deserialize the entry metadata.
     * @param in The input stream.
     */
    void deserialize(std::istream& in) {
      S aid;
      util::serialize::deserialize_one(in, aid);
      axis_id = AxId(aid);
      view.deserialize(in);
    }
  };

  /**
   * @brief Helper to resolve an x-coordinate to a local bin index using an axis.
   *
   * @param ax_id The ID of the axis to use.
   * @param x The x-coordinate.
   * @return The local bin index.
   */
  [[nodiscard]] inline S get_axis_index(AxId ax_id, const T& x) const {
    assert(ax_id.id() < axes_.size());
    return std::visit(
        [&](const auto& ax) -> S {
          using Type = std::decay_t<decltype(ax)>;
          if constexpr (std::is_same_v<Type, std::monostate>) {
            assert(false && "Attempted axis lookup on histogram without axis");
            return 0;
          } else {
            const S idx = ax.index(axis_data_, x);
            assert(idx < ax.n_bins());
            return idx;
          }
        },
        axes_[ax_id.id()]);
  }

  /**
   * @brief Internal helper to finalize the booking of a histogram.
   *
   * @param name Unique name for the histogram.
   * @param axis_id ID of the axis.
   * @param n_bins Number of bins.
   * @param n_values Number of values per bin.
   * @return The unique `HistogramId`.
   * @throws std::invalid_argument If the name already exists.
   */
  Id book_with_id(std::string_view name, AxId axis_id, S n_bins, S n_values) {
    for (const auto& existing : names_) {
      if (existing == name) {
        throw std::invalid_argument("HistogramRegistry: name already exists: " + std::string(name));
      }
    }

    const Id id{static_cast<S>(entries_.size())};
    names_.emplace_back(name);
    entries_.push_back({axis_id, View(data_, n_bins, n_values)});
    return id;
  }

  HistogramData<NT> data_;    //!< Physical storage for all bins.
  AxisData<T, S> axis_data_;  //!< Storage for axis edges and parameters.

  std::vector<Entry> entries_;           //!< Map of HistogramId to axis and view.
  std::vector<AxisVariant<T, S>> axes_;  //!< Registry of axis definitions.
  std::vector<std::string> names_;       //!< Registry of histogram names.
};

}  // namespace kakuhen::histogram