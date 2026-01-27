#pragma once

#include "kakuhen/histogram/axis_data.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <vector>

namespace kakuhen::histogram {

/**
 * @brief Enum defining the supported axis types for histogram binning.
 */
enum class AxisType : uint8_t {
  None = 0,     //!< No axis defined (monostate).
  Uniform = 1,  //!< Uniformly binned axis.
  Variable = 2  //!< Variably binned axis defined by explicit edges.
};

/**
 * @brief Converts an AxisType to its string representation.
 * @param axtype The AxisType to convert.
 * @return A string view of the axis type name.
 */
[[nodiscard]] constexpr std::string_view to_string(AxisType axtype) noexcept {
  switch (axtype) {
    case AxisType::None:
      return "None";
    case AxisType::Uniform:
      return "Uniform";
    case AxisType::Variable:
      return "Variable";
  }
  return "Unknown";
}

/**
 * @brief Metadata required to locate and identify an axis definition in shared storage.
 *
 * This structure encapsulates the state needed to reconstruct an axis view and
 * verify its compatibility with the coordinate and index traits.
 *
 * @tparam T The coordinate value type (e.g., double).
 * @tparam S The index type (e.g., uint32_t).
 */
template <typename T, typename S>
struct AxisMetadata {
  using value_type = T;
  using size_type = S;

  AxisType type = AxisType::None;  //!< The specific axis implementation type.
  S offset = 0;                    //!< Starting index within the global AxisData storage.
  S size = 0;                      //!< Number of elements (parameters or edges) in storage.
  S n_bins = 0;                    //!< Total number of bins (including Underflow/Overflow).
  S stride = 1;                    //!< Stride multiplier for multi-dimensional indexing.

  /**
   * @brief Serializes the metadata to an output stream.
   * @param out The destination output stream.
   * @param with_type If true, prepends type identifiers for T and S.
   */
  void serialize(std::ostream& out, bool with_type = false) const noexcept {
    if (with_type) {
      const int16_t T_tos = kakuhen::util::type::get_type_or_size<T>();
      kakuhen::util::serialize::serialize_one<int16_t>(out, T_tos);
      const int16_t S_tos = kakuhen::util::type::get_type_or_size<S>();
      kakuhen::util::serialize::serialize_one<int16_t>(out, S_tos);
    }
    kakuhen::util::serialize::serialize_one<uint8_t>(out, static_cast<uint8_t>(type));
    kakuhen::util::serialize::serialize_one<S>(out, offset);
    kakuhen::util::serialize::serialize_one<S>(out, size);
    kakuhen::util::serialize::serialize_one<S>(out, n_bins);
    kakuhen::util::serialize::serialize_one<S>(out, stride);
  }

  /**
   * @brief Deserializes the metadata from an input stream.
   * @param in The source input stream.
   * @param with_type If true, verifies type identifiers for T and S.
   * @throws std::runtime_error If type verification fails or the stream is corrupted.
   */
  void deserialize(std::istream& in, bool with_type = false) {
    if (with_type) {
      int16_t T_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, T_tos);
      if (T_tos != kakuhen::util::type::get_type_or_size<T>()) {
        throw std::runtime_error("AxisMetadata: coordinate type T mismatch.");
      }
      int16_t S_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, S_tos);
      if (S_tos != kakuhen::util::type::get_type_or_size<S>()) {
        throw std::runtime_error("AxisMetadata: index type S mismatch.");
      }
    }
    uint8_t t;
    kakuhen::util::serialize::deserialize_one<uint8_t>(in, t);
    type = static_cast<AxisType>(t);
    kakuhen::util::serialize::deserialize_one<S>(in, offset);
    kakuhen::util::serialize::deserialize_one<S>(in, size);
    kakuhen::util::serialize::deserialize_one<S>(in, n_bins);
    kakuhen::util::serialize::deserialize_one<S>(in, stride);
  }

  /**
   * @brief Compares two metadata objects for equality.
   */
  [[nodiscard]] bool operator==(const AxisMetadata& other) const noexcept {
    return type == other.type && offset == other.offset && size == other.size &&
           n_bins == other.n_bins && stride == other.stride;
  }

  /**
   * @brief Compares two metadata objects for inequality.
   */
  [[nodiscard]] bool operator!=(const AxisMetadata& other) const noexcept {
    return !(*this == other);
  }
};

/**
 * @brief CRTP base class for histogram axis views.
 *
 * AxisView provides the interface for mapping coordinates to bin indices.
 * It stores metadata describing where its parameters are located in a shared
 * AxisData storage.
 *
 * Indexing Convention:
 * - 0: Underflow (x < regular_range_min)
 * - 1 .. N: Regular bins
 * - N + 1: Overflow (x >= regular_range_max)
 *
 * @tparam Derived The derived axis type (UniformAxis or VariableAxis).
 * @tparam T The coordinate value type.
 * @tparam S The index type.
 */
template <typename Derived, typename T, typename S>
class AxisView {
 public:
  using value_type = T;
  using size_type = S;
  using metadata_type = AxisMetadata<T, S>;

  /**
   * @brief Constructs an AxisView from metadata.
   * @param meta The axis metadata.
   */
  explicit AxisView(const metadata_type& meta) : meta_{meta} {}

  /**
   * @brief Maps a coordinate to its corresponding bin index.
   * @param axis_data Shared storage containing the axis parameters.
   * @param x The coordinate value to map.
   * @return The bin index (0 to n_bins - 1) multiplied by the stride.
   */
  [[nodiscard]] S index(const AxisData<T, S>& axis_data, const T& x) const {
    return static_cast<const Derived*>(this)->index_impl(axis_data, x) * meta_.stride;
  }

  /**
   * @brief Access the underlying metadata.
   * @return A const reference to the metadata.
   */
  [[nodiscard]] const metadata_type& metadata() const noexcept {
    return meta_;
  }

  /**
   * @brief Get the total number of bins.
   * @return Bin count including underflow and overflow.
   */
  [[nodiscard]] S n_bins() const noexcept {
    return meta_.n_bins;
  }

  /**
   * @brief Get the offset into the shared AxisData storage.
   * @return The starting index.
   */
  [[nodiscard]] S offset() const noexcept {
    return meta_.offset;
  }

  /**
   * @brief Get the number of data points used by this axis in shared storage.
   * @return The element count.
   */
  [[nodiscard]] S size() const noexcept {
    return meta_.size;
  }

  /**
   * @brief Get the stride multiplier for this axis.
   * @return The stride value.
   */
  [[nodiscard]] S stride() const noexcept {
    return meta_.stride;
  }

  /**
   * @brief Sets the stride multiplier for this axis.
   * @param stride The new stride value.
   */
  void set_stride(S stride) noexcept {
    meta_.stride = stride;
  }

  /**
   * @brief Serializes the axis view metadata.
   * @param out The destination output stream.
   * @param with_type Whether to include type identifiers for verification.
   */
  void serialize(std::ostream& out, bool with_type = false) const noexcept {
    meta_.serialize(out, with_type);
  }

  /**
   * @brief Deserializes the axis view metadata.
   * @param in The source input stream.
   * @param with_type Whether to verify type identifiers.
   */
  void deserialize(std::istream& in, bool with_type = false) {
    meta_.deserialize(in, with_type);
  }

 protected:
  metadata_type meta_;  //!< Axis location and bin count information.
};

/**
 * @brief Represents an axis with uniformly spaced bins.
 *
 * Maps coordinates to indices in O(1) time using a linear transformation.
 *
 * @note Data layout in AxisData: `[min, max, scale]`
 *
 * @tparam T The coordinate value type.
 * @tparam S The index type.
 */
template <typename T, typename S>
class UniformAxis : public AxisView<UniformAxis<T, S>, T, S> {
 public:
  using Base = AxisView<UniformAxis<T, S>, T, S>;
  using metadata_type = typename Base::metadata_type;
  using Base::meta_;

  /**
   * @brief Constructs a UniformAxis from existing metadata.
   * @param meta The axis metadata.
   */
  explicit UniformAxis(const metadata_type& meta) : Base(meta) {
    assert(meta.type == AxisType::Uniform);
  }

  /**
   * @brief Constructs a UniformAxis and registers its parameters in AxisData.
   *
   * @param data Shared axis data storage.
   * @param n_bins Number of regular bins (total bins will be n_bins + 2).
   * @param min The lower bound of the first regular bin.
   * @param max The upper bound of the last regular bin.
   * @throws std::invalid_argument if n_bins is 0 or min >= max.
   */
  UniformAxis(AxisData<T, S>& data, S n_bins, const T& min, const T& max)
      : Base(validate_and_create(data, n_bins, min, max)) {}

  /**
   * @brief Maps a coordinate to a bin index using uniform mapping.
   * @param axis_data Shared storage containing the axis parameters.
   * @param x The coordinate value.
   * @return The bin index.
   */
  [[nodiscard]] S index_impl(const AxisData<T, S>& axis_data, const T& x) const noexcept {
    assert(meta_.type == AxisType::Uniform);
    const T& min_val = axis_data[meta_.offset];
    const T& max_val = axis_data[meta_.offset + 1];
    const T& scale_val = axis_data[meta_.offset + 2];

    if (x < min_val) return 0;                  // Underflow
    if (x >= max_val) return meta_.n_bins - 1;  // Overflow

    return static_cast<S>(1 + (x - min_val) * scale_val);
  }

 private:
  /**
   * @brief Validates parameters and appends data to storage before base initialization.
   */
  static metadata_type validate_and_create(AxisData<T, S>& data, S n_bins, const T& min,
                                           const T& max) {
    if (n_bins == 0) throw std::invalid_argument("UniformAxis: n_bins must be > 0");
    if (min >= max) throw std::invalid_argument("UniformAxis: min must be < max");
    return {AxisType::Uniform, data.add_data(min, max, static_cast<T>(n_bins) / (max - min)), S(3),
            static_cast<S>(n_bins + 2), 1};
  }
};

/**
 * @brief Represents an axis with variable bin widths.
 *
 * Maps coordinates to indices in O(log N) time using binary search over edges.
 *
 * @note Data layout in AxisData: `[edge_0, edge_1, ..., edge_N]`
 *
 * @tparam T The coordinate value type.
 * @tparam S The index type.
 */
template <typename T, typename S>
class VariableAxis : public AxisView<VariableAxis<T, S>, T, S> {
 public:
  using Base = AxisView<VariableAxis<T, S>, T, S>;
  using metadata_type = typename Base::metadata_type;
  using Base::meta_;

  /**
   * @brief Constructs a VariableAxis from existing metadata.
   * @param meta The axis metadata.
   */
  explicit VariableAxis(const metadata_type& meta) : Base(meta) {
    assert(meta.type == AxisType::Variable);
  }

  /**
   * @brief Constructs a VariableAxis and registers its edges in AxisData.
   *
   * @param data The shared axis data storage.
   * @param edges The bin edges (must be sorted).
   * @throws std::invalid_argument if edges are not sorted or fewer than 2.
   */
  VariableAxis(AxisData<T, S>& data, const std::vector<T>& edges)
      : Base(validate_and_create(data, edges)) {}

  /**
   * @brief Maps a coordinate to a bin index using binary search.
   * @param axis_data Shared storage containing the axis parameters.
   * @param x The coordinate value.
   * @return The bin index.
   */
  [[nodiscard]] S index_impl(const AxisData<T, S>& axis_data, const T& x) const {
    assert(meta_.type == AxisType::Variable);
    auto begin = axis_data.data().begin() + meta_.offset;
    auto end = begin + meta_.size;

    if (x < *begin) return 0;                      // Underflow
    if (x >= *(end - 1)) return meta_.n_bins - 1;  // Overflow

    auto it = std::upper_bound(begin, end, x);
    return static_cast<S>(std::distance(begin, it));
  }

 private:
  /**
   * @brief Validates edges and appends to storage before base initialization.
   */
  static metadata_type validate_and_create(AxisData<T, S>& data, const std::vector<T>& edges) {
    if (edges.size() < 2) throw std::invalid_argument("VariableAxis: requires at least 2 edges");
    if (!std::is_sorted(edges.begin(), edges.end())) {
      throw std::invalid_argument("VariableAxis: edges must be sorted");
    }
    return {AxisType::Variable, data.add_data(edges), static_cast<S>(edges.size()),
            static_cast<S>(edges.size() + 1), 1};
  }
};

/**
 * @brief Variant type holding any supported axis implementation.
 * @tparam T The coordinate value type.
 * @tparam S The index type.
 */
template <typename T, typename S>
using AxisVariant = std::variant<std::monostate, UniformAxis<T, S>, VariableAxis<T, S>>;

/**
 * @brief Factory function to restore an axis view from its metadata.
 *
 * @tparam T The value type.
 * @tparam S The index type.
 * @param meta The axis metadata.
 * @return An `AxisVariant` containing the restored axis view.
 */
template <typename T, typename S>
[[nodiscard]] AxisVariant<T, S> restore_axis(const AxisMetadata<T, S>& meta) {
  switch (meta.type) {
    case AxisType::Uniform:
      return UniformAxis<T, S>(meta);
    case AxisType::Variable:
      return VariableAxis<T, S>(meta);
    default:
      return std::monostate{};
  }
}

}  // namespace kakuhen::histogram
