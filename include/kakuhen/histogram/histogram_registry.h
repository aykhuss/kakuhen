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

    S n_bins = std::visit(
        [](const auto& ax) -> S {
          if constexpr (std::is_same_v<std::decay_t<decltype(ax)>, std::monostate>) {
            throw std::invalid_argument(
                "HistogramRegistry: cannot book with None axis using this method.");
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
   * @brief Fills a registered histogram with a range of values.
   */
  template <typename Buffer, typename Range>
  void fill(Buffer& buffer, Id id, S local_bin_idx, const Range& values) const {
    entries_[id.id()].view.fill(buffer, local_bin_idx, values);
  }

  /**
   * @brief Fills a registered histogram with a single value.
   */
  template <typename Buffer>
  void fill(Buffer& buffer, Id id, S local_bin_idx, const T& value) const {
    entries_[id.id()].view.fill(buffer, local_bin_idx, value);
  }

  /**
   * @brief Fills a registered histogram by mapping an x-coordinate to a bin index.
   */
  template <typename Buffer>
  void fill(Buffer& buffer, Id id, const T& x, const T& value) const {
    const auto& entry = entries_[id.id()];
    const auto& axis_var = axes_[entry.axis_id.id()];

    std::visit(
        [&](const auto& ax) {
          using Type = std::decay_t<decltype(ax)>;

          if constexpr (!std::is_same_v<Type, std::monostate>) {
            S bin_idx = ax.index(axis_data_, x);
            // with under-/over-flow bins, we always return a valid index
            assert(bin_idx < ax.n_bins());
            entry.view.fill(buffer, bin_idx, value);
          }
        },
        axis_var);
  }

  /**
   * @brief Flushes a buffer into the registry's global data storage.
   */
  template <typename Buffer>
  void flush(Buffer& buffer) {
    buffer.flush(data_);
  }

  /**
   * @brief Creates and initializes a thread-local buffer.
   * @tparam Acc The accumulator type to use.
   */
  template <typename Acc = kakuhen::util::accumulator::Accumulator<T>>
  [[nodiscard]] auto create_buffer() const {
    HistogramBuffer<NT, Acc> buffer;
    const S reserve_size = kakuhen::util::math::max(num_histograms(), num_axes());
    buffer.init(data_.size(), reserve_size);
    return buffer;
  }

  /// @name Accessors
  /// @{

  /**
   * @brief Access the underlying global bin storage.
   */
  [[nodiscard]] HistogramData<NT>& data() noexcept {
    return data_;
  }

  /**
   * @brief Access the underlying global bin storage (const).
   */
  [[nodiscard]] const HistogramData<NT>& data() const noexcept {
    return data_;
  }

  /**
   * @brief Access the underlying axis parameter storage.
   */
  [[nodiscard]] AxisData<T, S>& axis_data() noexcept {
    return axis_data_;
  }

  /**
   * @brief Access the underlying axis parameter storage (const).
   */
  [[nodiscard]] const AxisData<T, S>& axis_data() const noexcept {
    return axis_data_;
  }

  /**
   * @brief Retrieve the list of all registered histogram IDs.
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
   */
  [[nodiscard]] View get_view(Id id) const {
    return entries_[id.id()].view;
  }

  /**
   * @brief Retrieve the human-readable name for a specific histogram.
   */
  [[nodiscard]] std::string_view get_name(Id id) const {
    return names_[id.id()];
  }

  /**
   * @brief Look up a histogram's unique ID by its registered name.
   * @note This lookup is O(N).
   */
  [[nodiscard]] Id get_id(std::string_view name) const {
    for (size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) return Id{static_cast<S>(i)};
    }
    throw std::runtime_error("HistogramRegistry: not found: " + std::string(name));
  }

  /**
   * @brief Returns the total number of registered histograms.
   */
  [[nodiscard]] S num_histograms() const noexcept {
    return static_cast<S>(entries_.size());
  }

  /**
   * @brief Returns the total number of registered axis definitions.
   */
  [[nodiscard]] S num_axes() const noexcept {
    return static_cast<S>(axes_.size());
  }

  /// @}

  /// @name Serialization
  /// @{

  /**
   * @brief Serializes the entire registry state to an output stream.
   *
   * @param out The output stream.
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
   * @param in The input stream.
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

    void serialize(std::ostream& out) const noexcept {
      util::serialize::serialize_one(out, axis_id.id());
      view.serialize(out);
    }

    void deserialize(std::istream& in) {
      S aid;
      util::serialize::deserialize_one(in, aid);
      axis_id = AxId(aid);
      view.deserialize(in);
    }
  };

  /**
   * @brief Internal helper to finalize the booking of a histogram.
   * @throws std::invalid_argument If the name already exists.
   */
  Id book_with_id(std::string_view name, AxId axis_id, S n_bins, S n_values) {
    const std::string s_name(name);
    for (const auto& existing : names_) {
      if (existing == s_name) {
        throw std::invalid_argument("HistogramRegistry: name already exists: " + s_name);
      }
    }

    const Id id{static_cast<S>(entries_.size())};
    names_.emplace_back(s_name);
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
