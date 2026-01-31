#pragma once

#include "kakuhen/histogram/axis_data.h"
#include "kakuhen/histogram/axis_id.h"
#include "kakuhen/histogram/axis_view.h"
#include "kakuhen/histogram/bin_range.h"
#include "kakuhen/histogram/histogram_buffer.h"
#include "kakuhen/histogram/histogram_data.h"
#include "kakuhen/histogram/histogram_id.h"
#include "kakuhen/histogram/histogram_view.h"
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
 * This class provides a centralized point for booking and filling histograms,
 * ensuring that memory allocation and multi-dimensional axis mapping are handled
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
  using AxisVar = AxisViewVariant<T, S>;

  /**
   * @brief Books a new histogram without an associated axis (view only).
   *
   * @param name A unique identifier for the histogram.
   * @param n_values_per_bin The number of values per bin (default 1).
   * @param n_bins The number of bins to allocate.
   * @return A `HistogramId` handle.
   * @throws std::invalid_argument If the name is already in use.
   */
  [[nodiscard]] Id book(std::string_view name, S n_values_per_bin, S n_bins) {
    axes_.emplace_back(std::monostate{});
    AxId ax_id{static_cast<S>(axes_.size() - 1), 1};
    return book_with_id(name, ax_id, n_bins, n_values_per_bin);
  }

  /**
   * @brief Books a new histogram by providing one or more self-contained Axis objects.
   *
   * The provided axis objects are duplicated into the registry's internal storage,
   * and strides are automatically calculated for multi-dimensional indexing.
   *
   * @tparam FirstAxis Type of the first Axis object.
   * @tparam AxisTypes Types of the subsequent Axis objects.
   * @param name Unique name for the histogram.
   * @param n_values_per_bin Number of values per bin.
   * @param first The first Axis object.
   * @param rest Subsequent Axis objects.
   * @return A `HistogramId` handle.
   * @throws std::invalid_argument If the name is already in use.
   */
  template <typename FirstAxis, typename... AxisTypes>
  [[nodiscard]] Id book(std::string_view name, S n_values_per_bin, const FirstAxis& first,
                        const AxisTypes&... rest)
    requires(!std::is_integral_v<std::decay_t<FirstAxis>>)
  {
    const S start_index = static_cast<S>(axes_.size());
    const S ndim = static_cast<S>(sizeof...(AxisTypes) + 1);

    // Duplicate each axis into registry and store the resulting view
    axes_.push_back(first.duplicate(axis_data_));
    (axes_.push_back(rest.duplicate(axis_data_)), ...);

    // Calculate and set strides in reverse order (row-major like logic)
    S current_stride = 1;
    for (S i = start_index + ndim; i-- > start_index;) {
      std::visit(
          [&current_stride](auto&& ax) {
            using Type = std::decay_t<decltype(ax)>;
            if constexpr (!std::is_same_v<Type, std::monostate>) {
              ax.set_stride(current_stride);
              current_stride *= ax.n_bins();
            }
          },
          axes_[i]);
    }

    return book_with_id(name, AxId{start_index, ndim}, current_stride, n_values_per_bin);
  }

  /// @name Filling
  /// @{

  /**
   * @brief Fills a registered histogram with a span of values using a local bin index.
   *
   * @tparam Buffer The type of the histogram buffer.
   * @param buffer The thread-local buffer to fill.
   * @param id The ID of the histogram.
   * @param values The span of values to accumulate.
   * @param local_bin_idx The index of the bin within the histogram.
   */
  template <typename Buffer>
  void fill_by_index(Buffer& buffer, Id id, std::span<const T> values, S local_bin_idx) const {
    assert(id.id() < entries_.size());
    entries_[id.id()].view.fill_by_index(buffer, values, local_bin_idx);
  }

  /**
   * @brief Fills a registered histogram with a single value using a local bin index.
   *
   * @tparam Buffer The type of the histogram buffer.
   * @param buffer The thread-local buffer to fill.
   * @param id The ID of the histogram.
   * @param value The value to accumulate.
   * @param local_bin_idx The index of the bin within the histogram.
   */
  template <typename Buffer>
  void fill_by_index(Buffer& buffer, Id id, const T& value, S local_bin_idx) const {
    assert(id.id() < entries_.size());
    entries_[id.id()].view.fill_by_index(buffer, value, local_bin_idx);
  }

  /**
   * @brief Fills a registered histogram by mapping multi-dimensional coordinates to a bin index,
   * providing a span of values.
   *
   * @tparam Buffer The type of the histogram buffer.
   * @tparam Coords Coordinate types.
   * @param buffer The thread-local buffer to fill.
   * @param id The ID of the histogram.
   * @param values The span of values to accumulate.
   * @param x Coordinates for each dimension.
   */
  template <typename Buffer, typename... Coords>
  void fill(Buffer& buffer, Id id, std::span<const T> values, Coords&&... x) const {
    assert(id.id() < entries_.size());
    const auto& entry = entries_[id.id()];
    assert(sizeof...(x) == entry.axis_id.ndim());
    const S bin_idx = compute_index<0>(entry.axis_id, std::forward<Coords>(x)...);
    entry.view.fill_by_index(buffer, values, bin_idx);
  }

  /**
   * @brief Fills a registered histogram by mapping multi-dimensional coordinates to a bin index,
   * providing a single value.
   *
   * @tparam Buffer The type of the histogram buffer.
   * @tparam Coords Coordinate types.
   * @param buffer The thread-local buffer to fill.
   * @param id The ID of the histogram.
   * @param value The value to accumulate.
   * @param x Coordinates for each dimension.
   */
  template <typename Buffer, typename... Coords>
  void fill(Buffer& buffer, Id id, const T& value, Coords&&... x) const {
    assert(id.id() < entries_.size());
    const auto& entry = entries_[id.id()];
    assert(sizeof...(x) == entry.axis_id.ndim());
    const S bin_idx = compute_index<0>(entry.axis_id, std::forward<Coords>(x)...);
    entry.view.fill_by_index(buffer, value, bin_idx);
  }

  /// @}

  /**
   * @brief Flushes a buffer into the registry's global data storage.
   *
   * @tparam Buffer The type of the histogram buffer.
   * @param buffer The buffer to flush.
   */
  template <typename Buffer>
  void flush(Buffer& buffer) noexcept(noexcept(buffer.flush(data_))) {
    buffer.flush(data_);
  }

  /**
   * @brief Creates and initializes a thread-local buffer for filling histograms.
   *
   * @tparam Acc The accumulator type to use (default: TwoSum via
   * kakuhen::util::accumulator::Accumulator).
   * @return A newly initialized `HistogramBuffer`.
   */
  template <typename Acc = kakuhen::util::accumulator::Accumulator<T>>
  [[nodiscard]] auto create_buffer() const {
    HistogramBuffer<NT, Acc> buffer;
    const S reserve_size = num_entries();
    buffer.init(data_.size(), reserve_size);
    return buffer;
  }

  /**
   * @brief Prints a summary of all registered histograms using the provided printer.
   *
   * @tparam Printer The type of the printer (e.g., JSONPrinter).
   * @param prt The printer instance to use.
   */
  template <typename Printer>
  void print(Printer& prt) const {
    prt.reset();
    prt.global_header(*this);
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      prt.histogram_header(*this);
      prt.histogram_row(*this, entries_[i], i);
      prt.histogram_footer(*this);
    }
    prt.global_footer(*this);
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
   * @brief Returns the total number of registered histograms.
   */
  [[nodiscard]] S num_entries() const noexcept {
    return static_cast<S>(entries_.size());
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
   * @brief Look up a histogram's unique ID by its registered name.
   * @throws std::runtime_error If the name is not found.
   */
  [[nodiscard]] Id get_id(std::string_view name) const {
    for (size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) return Id{static_cast<S>(i)};
    }
    throw std::runtime_error("HistogramRegistry: not found: " + std::string(name));
  }

  /**
   * @brief Retrieve the human-readable name for a specific histogram.
   */
  [[nodiscard]] std::string_view get_name(Id id) const noexcept {
    assert(id.id() < names_.size());
    return names_[id.id()];
  }

  /**
   * @brief Retrieve the view handle for a specific histogram.
   */
  [[nodiscard]] View get_view(Id id) const noexcept {
    assert(id.id() < entries_.size());
    return entries_[id.id()].view;
  }

  /**
   * @brief Get the number of dimensions for a specific histogram.
   *
   * @param id The histogram ID.
   * @return The number of dimensions.
   */
  [[nodiscard]] S get_ndim(Id id) const noexcept {
    assert(id.id() < entries_.size());
    return entries_[id.id()].axis_id.ndim();
  }

  [[nodiscard]] S get_nbins(Id id) const noexcept {
    assert(id.id() < entries_.size());
    return entries_[id.id()].view.n_bins();
  }

  [[nodiscard]] S get_nvalues(Id id) const noexcept {
    assert(id.id() < entries_.size());
    return entries_[id.id()].view.stride();
  }

  [[nodiscard]] auto get_bin_ranges(Id id) const noexcept {
    std::vector<std::vector<BinRange<T>>> result;
    result.reserve(get_ndim(id));
    const AxId& ax = entries_[id.id()].axis_id;
    for (S i = ax.id(); i < ax.id() + ax.ndim(); ++i) {
      result.push_back(std::visit(
          [&](const auto& iax) {
            using Type = std::decay_t<decltype(ax)>;
            if constexpr (std::is_same_v<Type, std::monostate>) {
              assert(false && "Attempted axis lookup on histogram without axis");
              return std::vector<BinRange<T>>();
            } else {
              return ax.bin_ranges(axis_data_);
            }
          },
          axes_[i]));
    }
    return result;
  }

  /**
   * @brief Access the accumulator for a specific bin in a registered histogram.
   *
   * @param id The histogram ID.
   * @param bin_idx The local flattened bin index.
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
   * @brief Get the mean value (sum of weights / N) for a specific bin.
   */
  [[nodiscard]] T get_bin_value(Id id, S bin_idx, S value_idx = 0) const noexcept {
    const auto& bin = get_bin_noexcept(id, bin_idx, value_idx);
    const T n = static_cast<T>(data_.count());
    if (n == T(0)) return T(0);
    return bin.weight() / n;
  }

  /**
   * @brief Get the variance of the mean value for a specific bin.
   */
  [[nodiscard]] T get_bin_variance(Id id, S bin_idx, S value_idx = 0) const noexcept {
    const U n_count = data_.count();
    if (n_count <= 1) return T(0);
    const auto& bin = get_bin_noexcept(id, bin_idx, value_idx);
    const T n = static_cast<T>(n_count);
    const T mean = bin.weight() / n;
    return (bin.weight_sq() / n - mean * mean) / (n - T(1));
  }

  /**
   * @brief Get the statistical error (standard deviation of the mean) for a specific bin.
   */
  [[nodiscard]] T get_bin_error(Id id, S bin_idx, S value_idx = 0) const noexcept {
    return std::sqrt(get_bin_variance(id, bin_idx, value_idx));
  }

  /// @}

  /// @name Serialization
  /// @{

  /**
   * @brief Serializes the entire registry state to an output stream.
   *
   * @param out The destination output stream.
   * @param with_type Whether to prepend type identifiers for verification.
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
    data_.serialize(out);
    axis_data_.serialize(out);
    util::serialize::serialize_size(out, names_.size());
    for (const auto& name : names_)
      util::serialize::serialize_one(out, name);
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
    util::serialize::serialize_size(out, entries_.size());
    for (const auto& entry : entries_)
      entry.serialize(out);
  }

  /**
   * @brief Deserializes the entire registry state from an input stream.
   *
   * @param in The source input stream.
   * @param with_type Whether to verify type identifiers.
   * @throws std::runtime_error If type mismatch or stream corruption occurs.
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
    data_.deserialize(in);
    axis_data_.deserialize(in);
    std::size_t n_names;
    util::serialize::deserialize_size(in, n_names);
    names_.resize(n_names);
    for (auto& name : names_)
      util::serialize::deserialize_one(in, name);
    std::size_t n_axes;
    util::serialize::deserialize_size(in, n_axes);
    axes_.clear();
    axes_.reserve(n_axes);
    for (std::size_t i = 0; i < n_axes; ++i) {
      AxisMetadata<T, S> meta;
      meta.deserialize(in);
      axes_.emplace_back(restore_axis(meta));
    }
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

  /**
   * @brief Resets the registry, clearing all histograms, axes, and data.
   */
  void clear() noexcept {
    data_.clear();
    axis_data_.clear();
    entries_.clear();
    axes_.clear();
    names_.clear();
  }

 private:
  /**
   * @brief Internal structure mapping a histogram handle to its components.
   */
  struct Entry {
    AxId axis_id{0, 0};  //!< The associated set of axes.
    View view{0, 0, 0};  //!< Handle to the physical bin storage.

    void serialize(std::ostream& out) const noexcept {
      util::serialize::serialize_one(out, axis_id.id());
      util::serialize::serialize_one(out, axis_id.ndim());
      view.serialize(out);
    }

    void deserialize(std::istream& in) {
      S aid, andim;
      util::serialize::deserialize_one(in, aid);
      util::serialize::deserialize_one(in, andim);
      axis_id = AxId(aid, andim);
      view.deserialize(in);
    }
  };

  /**
   * @brief Resolves an x-coordinate to a bin index using a specific registered axis.
   */
  [[nodiscard]] inline S get_axis_index(AxId ax_id, const T& x) const noexcept {
    assert(ax_id.id() < axes_.size());
    return std::visit(
        [&](const auto& ax) -> S {
          using Type = std::decay_t<decltype(ax)>;
          if constexpr (std::is_same_v<Type, std::monostate>) {
            assert(false && "Attempted axis lookup on histogram without axis");
            return 0;
          } else {
            return ax.index(axis_data_, x);
          }
        },
        axes_[ax_id.id()]);
  }

  /**
   * @brief Internal helper to finalize the booking of a histogram and store its metadata.
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

  /**
   * @brief Recursively computes the flattened multi-dimensional bin index from coordinates.
   */
  template <S Dim, typename Arg, typename... Rest>
  [[nodiscard]] S compute_index(AxId ax_id, Arg&& arg, Rest&&... rest) const noexcept {
    static_assert(Dim < 32, "Suspicious histogram dimensionality (>= 32)");
    static_assert(std::is_convertible_v<std::decay_t<Arg>, T>,
                  "Coordinate must be convertible to the axis value type T");
    const S local_idx = get_axis_index(AxId(ax_id.id() + Dim, 1), static_cast<T>(arg));
    if constexpr (sizeof...(rest) > 0) {
      return local_idx + compute_index<Dim + 1>(ax_id, std::forward<Rest>(rest)...);
    } else {
      return local_idx;
    }
  }

  /**
   * @brief Internal helper for getting a bin without range checks (use with caution).
   */
  [[nodiscard]] const auto& get_bin_noexcept(Id id, S bin_idx, S value_idx = 0) const noexcept {
    assert(id.id() < entries_.size());
    const auto& view = entries_[id.id()].view;
    assert(bin_idx < view.n_bins());
    assert(value_idx < view.stride());
    return view.get_bin(data_, bin_idx, value_idx);
  }

  HistogramData<NT> data_;    //!< Physical storage for all accumulated bins.
  AxisData<T, S> axis_data_;  //!< Shared storage for axis parameters/edges.

  std::vector<Entry> entries_;               //!< Registered histogram metadata.
  std::vector<AxisViewVariant<T, S>> axes_;  //!< Registered axis definitions.
  std::vector<std::string> names_;           //!< Unique names of registered histograms.
};

}  // namespace kakuhen::histogram
