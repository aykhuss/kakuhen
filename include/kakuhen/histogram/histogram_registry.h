#include "kakuhen/histogram/axis_data.h"
#include "kakuhen/histogram/axis_id.h"
#include "kakuhen/histogram/axis_view.h"
#include "kakuhen/histogram/histogram_buffer.h"
#include "kakuhen/histogram/histogram_data.h"
#include "kakuhen/histogram/histogram_id.h"
#include "kakuhen/histogram/histogram_view.h"
#include "kakuhen/util/numeric_traits.h"
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace kakuhen::histogram {

/*!
 * @brief Manages the lifecycle and registration of multiple histograms.
 *
 * The `HistogramRegistry` serves as a facade over `HistogramData` and `AxisData`.
 * It manages:
 * 1. Global bin storage (HistogramData)
 * 2. Global axis definition storage (AxisData)
 * 3. Registered Axes (list of AxisViews)
 * 4. Registered Histograms (mapping of Name -> HistogramView + AxisId)
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
  using AxId = AxisId<S>;

  // Unified Axis View Type
  using AxisVariant = std::variant<std::monostate, UniformAxis<T, S>, VariableAxis<T, S>>;

  /*!
   * @brief Creates and registers a new axis.
   *
   * @tparam AxisType The type of axis (UniformAxis or VariableAxis).
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

  /*!
   * @brief Creates and registers a new axis using an initializer list.
   * Useful for VariableAxis edges.
   */
  template <typename AxisType>
  [[nodiscard]] AxId create_axis(std::initializer_list<T> list)
    requires(std::is_same_v<AxisType, VariableAxis<T, S>>)
  {
    return create_axis<AxisType>(std::vector<T>(list));
  }

  /*!
   * @brief Books a new histogram using an existing axis ID.
   *
   * @param name A unique identifier for the histogram.
   * @param axis_id The ID of the registered axis.
   * @param n_values_per_bin The number of values per bin.
   * @return A `HistogramId` handle.
   */
  [[nodiscard]] Id book(std::string_view name, AxId axis_id, S n_values_per_bin = 1) {
    if (axis_id.id() >= axes_.size()) {
      throw std::out_of_range("Invalid AxisId");
    }

    const auto& axis_var = axes_[axis_id.id()];

    S n_bins = std::visit(
        [](const auto& ax) -> S {
          if constexpr (std::is_same_v<std::decay_t<decltype(ax)>, std::monostate>) {
            throw std::invalid_argument(
                "Cannot book histogram with monostate axis using this method.");
          } else {
            return ax.n_bins();
          }
        },
        axis_var);

    return book_with_id(name, axis_id, n_bins, n_values_per_bin);
  }

  /*!
   * @brief Books a new histogram (View only, no axis).
   *
   * @param name A unique identifier for the histogram.
   * @param n_bins The number of bins.
   * @param n_values_per_bin The number of values per bin (default 1).
   * @return A `HistogramId` handle to the booked histogram.
   */
  [[nodiscard]] Id book(std::string_view name, S n_bins, S n_values_per_bin = 1) {
    axes_.emplace_back(std::monostate{});
    AxId ax_id{static_cast<S>(axes_.size() - 1)};
    return book_with_id(name, ax_id, n_bins, n_values_per_bin);
  }

  /*!
   * @brief Fills a registered histogram with a range of values (by local bin Index).
   */
  template <typename Buffer, typename Range>
  void fill(Buffer& buffer, Id id, S local_bin_idx, const Range& values) const {
    entries_[id.id()].view.fill(buffer, local_bin_idx, values);
  }

  /*!
   * @brief Fills a registered histogram with a single value (by local bin Index).
   */
  template <typename Buffer>
  void fill(Buffer& buffer, Id id, S local_bin_idx, const T& value) const {
    entries_[id.id()].view.fill(buffer, local_bin_idx, value);
  }

  /*!
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

            if (bin_idx < ax.n_bins()) {
              entry.view.fill(buffer, bin_idx, value);
            }
          }
        },
        axis_var);
  }

  /*!
   * @brief Flushes a buffer into the registry's data.
   */
  template <typename Buffer>
  void flush(Buffer& buffer) {
    buffer.flush(data_);
  }

  /*!
   * @brief Creates and initializes a thread-local buffer.
   */
  template <typename Acc = kakuhen::util::accumulator::Accumulator<T>>
  auto create_buffer(S reserve_size = 1024) const {
    HistogramBuffer<NT, Acc> buffer;
    buffer.init(data_.size(), reserve_size);
    return buffer;
  }

  HistogramData<NT>& data() noexcept {
    return data_;
  }
  const HistogramData<NT>& data() const noexcept {
    return data_;
  }

  AxisData<T, S>& axis_data() noexcept {
    return axis_data_;
  }
  const AxisData<T, S>& axis_data() const noexcept {
    return axis_data_;
  }

  std::vector<Id> ids() const {
    std::vector<Id> result;
    result.reserve(entries_.size());
    for (S i = 0; i < static_cast<S>(entries_.size()); ++i) {
      result.emplace_back(i);
    }
    return result;
  }

  View get_view(Id id) const {
    return entries_[id.id()].view;
  }
  std::string_view get_name(Id id) const {
    return names_[id.id()];
  }

  Id get_id(std::string_view name) const {
    for (size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) return Id{static_cast<S>(i)};
    }
    throw std::runtime_error("Histogram not found: " + std::string(name));
  }

 private:
  struct Entry {
    AxId axis_id;
    View view;
  };

  Id book_with_id(std::string_view name, AxId axis_id, S n_bins, S n_values) {
    names_.emplace_back(name);
    entries_.push_back({axis_id, View(data_, n_bins, n_values)});
    return Id{static_cast<S>(entries_.size() - 1)};
  }

  HistogramData<NT> data_;
  AxisData<T, S> axis_data_;

  std::vector<Entry> entries_;
  std::vector<AxisVariant> axes_;
  std::vector<std::string> names_;
};

}  // namespace kakuhen::histogram
