#pragma once

#include "kakuhen/histogram/axis_data.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace kakuhen::histogram {

/*!
 * @brief CRTP base class for histogram axes.
 *
 * Enforces the interface for mapping a coordinate `x` to a bin index.
 * Stores a reference to the shared AxisData and an offset to the data.
 *
 * @tparam Derived The derived axis type.
 * @tparam T The coordinate value type.
 * @tparam S The index type.
 */
template <typename Derived, typename T, typename S>
class AxisView {
 public:
  using value_type = T;
  using size_type = S;
  // using data_type = AxisData<T, S>;

  AxisView(AxisData<T, S>& axis_data, S offset, S size, S n_bins)
      : offset_{offset}, size_{size}, n_bins_{n_bins} {}

  /*!
   * @brief Maps a coordinate to a bin index.
   *
   * @param x The coordinate value.
   * @return The bin index, or -1 for underflow, size() for overflow.
   */
  [[nodiscard]] int index(AxisData<T, S>& axis_data, const T& x) const {
    return static_cast<const Derived*>(this)->index_impl(axis_data, x);
  }

  /*!
   * @brief Get the number of bins.
   */
  [[nodiscard]] S n_bins() const noexcept {
    return n_bins_;
  }

 protected:
  S offset_;
  S size_;
  S n_bins_;
};

/*!
 * @brief Represents a uniform binning axis.
 *
 * Maps a continuous value `x` to a bin index using a linear transformation.
 * Computation is O(1).
 * Stores {min, max, scale} in AxisData.
 */
template <typename T, typename S>
class UniformAxis : public AxisView<UniformAxis<T, S>, T, S> {
 public:
  using Base = AxisView<UniformAxis<T, S>, T, S>;
  using Base::n_bins_;
  using Base::offset_;
  using Base::size_;

  UniformAxis(AxisData<T, S>& data, S n_bins, const T& min, const T& max)
      : Base(data, data.add_data(min, max, static_cast<T>(n_bins) / (max - min)), S(3), n_bins) {
    if (n_bins == 0) throw std::invalid_argument("n_bins must be > 0");
    if (min >= max) throw std::invalid_argument("min must be < max");
  }

  [[nodiscard]] int index_impl(AxisData<T, S>& axis_data, const T& x) const noexcept {
    // Data layout: [min, max, scale]
    const T& min_val = axis_data[offset_];
    const T& max_val = axis_data[offset_ + 1];
    const T& scale_val = axis_data[offset_ + 2];

    if (x < min_val) return -1;
    if (x >= max_val) return static_cast<int>(n_bins_);
    return static_cast<int>((x - min_val) * scale_val);
  }
};

/*!
 * @brief Represents a variable binning axis.
 *
 * Maps a continuous value `x` to a bin index using binary search over edges.
 * Computation is O(log N).
 * Stores edges in AxisData.
 */
template <typename T, typename S>
class VariableAxis : public AxisView<VariableAxis<T, S>, T, S> {
 public:
  using Base = AxisView<VariableAxis<T, S>, T, S>;
  using Base::n_bins_;
  using Base::offset_;
  using Base::size_;

  /*!
   * @brief Constructs a variable binning axis.
   *
   * @param data The shared axis data.
   * @param edges The bin edges (must be sorted).
   */
  VariableAxis(AxisData<T, S>& data, const std::vector<T>& edges)
      : Base(data, data.add_data(edges), static_cast<S>(edges.size()),
             static_cast<S>(edges.size() - 1)) {
    if (edges.size() < 2) throw std::invalid_argument("VariableAxis requires at least 2 edges");
    if (!std::is_sorted(edges.begin(), edges.end())) {
      throw std::invalid_argument("Edges must be sorted");
    }
  }

  [[nodiscard]] int index_impl(AxisData<T, S>& axis_data, const T& x) const {
    auto begin = axis_data.data().begin() + offset_;
    auto end = begin + size_;

    if (x < *begin) return -1;
    if (x >= *(end - 1)) return static_cast<int>(n_bins_);

    auto it = std::upper_bound(begin, end, x);
    return static_cast<int>(std::distance(begin, it) - 1);
  }
};

}  // namespace kakuhen::histogram
