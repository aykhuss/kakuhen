#pragma once

#include "kakuhen/histogram/axis_data.h"
#include "kakuhen/histogram/axis_view.h"
#include <vector>

namespace kakuhen::histogram {

/**
 * @brief A self-contained axis that owns its data.
 *
 * Unlike AxisView which points to shared storage, Axis manages its own
 * parameter/edge storage internally. It wraps a concrete AxisView type
 * (UniformAxis or VariableAxis) and provides a simplified interface.
 *
 * This class is ideal for use as a building block for defining histograms
 * before they are registered in a shared HistogramRegistry.
 *
 * @tparam ConcreteAxis The specific axis implementation (UniformAxis or VariableAxis).
 * @tparam T The coordinate value type.
 * @tparam S The index type.
 */
template <template <typename, typename> class ConcreteAxis, typename T, typename S>
class Axis {
 public:
  using view_type = ConcreteAxis<T, S>;
  using value_type = T;
  using size_type = S;

  /**
   * @brief Constructs an Axis from an initializer list (e.g. for VariableAxis).
   * @param list Initializer list of values (e.g. bin edges).
   */
  explicit Axis(std::initializer_list<T> list) : data_(), view_(data_, std::vector<T>(list)) {}

  /**
   * @brief Constructs an Axis by forwarding arguments to the concrete axis view.
   *
   * The arguments are used to populate the internal AxisData and initialize the view.
   *
   * @tparam Args Constructor arguments for the concrete axis (excluding AxisData).
   * @param args Arguments forwarded to the ConcreteAxis constructor.
   */
  template <typename... Args>
  explicit Axis(Args&&... args) : data_(), view_(data_, std::forward<Args>(args)...) {}

  /**
   * @brief Maps a coordinate to a bin index.
   *
   * @param x The coordinate value.
   * @return The bin index (including UF/OF).
   */
  [[nodiscard]] S index(const T& x) const noexcept {
    return view_.index(data_, x);
  }

  /**
   * @brief Returns the bin edges of the regular bins.
   * @return A vector containing the regular bin boundaries.
   */
  [[nodiscard]] std::vector<T> edges() const {
    return view_.edges(data_);
  }

  /**
   * @brief Returns the full set of bin ranges for this axis.
   * @return A vector of BinRange objects for each bin in the axis.
   */
  [[nodiscard]] std::vector<BinRange<T>> bin_ranges() const {
    return view_.bin_ranges(data_);
  }

  /**
   * @brief Get the underlying axis view.
   * @return A const reference to the view.
   */
  [[nodiscard]] const view_type& view() const noexcept {
    return view_;
  }

  /**
   * @brief Get the total number of bins.
   * @return Bin count (including UF/OF).
   */
  [[nodiscard]] S n_bins() const noexcept {
    return view_.n_bins();
  }

  /**
   * @brief Get the internal data storage.
   * @return A const reference to the AxisData.
   */
  [[nodiscard]] const AxisData<T, S>& data() const noexcept {
    return data_;
  }

  /**
   * @brief Duplicates the axis into an external AxisData storage.
   *
   * This facilitates transferring a self-contained axis definition into
   * shared registry storage.
   *
   * @param target_data The external storage to receive the parameters/edges.
   * @return A new AxisView pointing to the duplicated data in target_data.
   */
  [[nodiscard]] view_type duplicate(AxisData<T, S>& target_data) const {
    const S new_offset = target_data.add_data(data_.data());
    auto meta = view_.metadata();
    meta.offset = new_offset;
    return view_type(meta);
  }

 private:
  AxisData<T, S> data_;  //!< Self-contained storage for axis parameters.
  view_type view_;       //!< The view operating on the internal data.
};

/**
 * @brief Convenience alias for a self-contained UniformAxis.
 */
template <typename T = double, typename S = uint32_t>
using Uniform = Axis<UniformAxis, T, S>;

/**
 * @brief Convenience alias for a self-contained VariableAxis.
 */
template <typename T = double, typename S = uint32_t>
using Variable = Axis<VariableAxis, T, S>;

}  // namespace kakuhen::histogram
