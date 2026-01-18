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
   * @brief Books a new histogram (View only, no axis).
   */
  [[nodiscard]] Id book(std::string_view name, S n_bins, S n_values_per_bin = 1) {
    return book_impl(name, std::monostate{}, n_bins, n_values_per_bin);
  }

  /*!
   * @brief Books a new histogram with an axis.
   *
   * @param name Unique name.
   * @param axis The axis view object (constructed with registry.axis_data()).
   */
  template <typename AxisType>
  [[nodiscard]] Id book(std::string_view name, AxisType axis, S n_values_per_bin = 1)
    requires(!std::is_integral_v<AxisType>)
  {
    S n_bins = axis.n_bins();  // Capture size before move
    return book_impl(name, std::move(axis), n_bins, n_values_per_bin);
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
            // Safe implementation:
            // We need to pass axis_data_ to index.
            // AxisView::index signature expects AxisData&, but we are const.
            // Since we know index() doesn't mutate data (based on implementation),
            // const_cast is acceptable here to bridge the API mismatch.
            int bin_idx = ax.index(const_cast<AxisData<T, S>&>(axis_data_), x);

            if (bin_idx >= 0 && static_cast<S>(bin_idx) < ax.n_bins()) {
              entry.view.fill(buffer, static_cast<S>(bin_idx), value);
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

  Id book_impl(std::string_view name, AxisVariant axis, S n_bins, S n_values) {
    names_.emplace_back(name);
    // Register axis
    axes_.emplace_back(std::move(axis));
    AxId ax_id{static_cast<S>(axes_.size() - 1)};

    // Register histogram
    entries_.push_back({ax_id, View(data_, n_bins, n_values)});
    return Id{static_cast<S>(entries_.size() - 1)};
  }

  HistogramData<NT> data_;
  AxisData<T, S> axis_data_;

  std::vector<Entry> entries_;
  std::vector<AxisVariant> axes_;
  std::vector<std::string> names_;
};

}  // namespace kakuhen::histogram
