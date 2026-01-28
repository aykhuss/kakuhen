#pragma once

#include "kakuhen/histogram/axis_data.h"
#include <algorithm>
#include <cassert>
#include <span>
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
 * @brief Policy for underflow and overflow bins.
 */
enum class UOFlowPolicy : uint8_t {
  None = 0,                                //!< No underflow or overflow bins.
  Underflow = 1 << 0,                      //!< Only underflow bin included.
  Overflow = 1 << 1,                       //!< Only overflow bin included.
  Both = Underflow | Overflow              //!< Both underflow and overflow bins included.
};

/**
 * @brief Checks if the policy includes an underflow bin.
 * @param flow The policy to check.
 * @return True if underflow is included.
 */
[[nodiscard]] constexpr bool has_underflow(UOFlowPolicy flow) noexcept {
  return static_cast<uint8_t>(flow) & static_cast<uint8_t>(UOFlowPolicy::Underflow);
}

/**
 * @brief Checks if the policy includes an overflow bin.
 * @param flow The policy to check.
 * @return True if overflow is included.
 */
[[nodiscard]] constexpr bool has_overflow(UOFlowPolicy flow) noexcept {
  return static_cast<uint8_t>(flow) & static_cast<uint8_t>(UOFlowPolicy::Overflow);
}

/**
 * @brief Returns the number of flow bins (0, 1, or 2) for a given policy.
 * @param flow The policy to check.
 * @return The number of extra bins allocated for flow.
 */
[[nodiscard]] constexpr uint8_t flow_count(UOFlowPolicy flow) noexcept {
  return (has_underflow(flow) ? 1 : 0) + (has_overflow(flow) ? 1 : 0);
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

  AxisType type = AxisType::None;          //!< The specific axis implementation type.
  UOFlowPolicy flow = UOFlowPolicy::Both;  //!< UF/OF bin configuration.
  S offset = 0;                            //!< Starting index within the global AxisData storage.
  S size = 0;                              //!< Number of elements (parameters or edges) in storage.
  S n_bins = 0;  //!< Total number of bins (including UF/OF as configured).
  S stride = 1;  //!< Stride multiplier for multi-dimensional indexing.

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
    kakuhen::util::serialize::serialize_one<uint8_t>(out, static_cast<uint8_t>(flow));
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
    uint8_t t, f;
    kakuhen::util::serialize::deserialize_one<uint8_t>(in, t);
    type = static_cast<AxisType>(t);
    kakuhen::util::serialize::deserialize_one<uint8_t>(in, f);
    flow = static_cast<UOFlowPolicy>(f);
    kakuhen::util::serialize::deserialize_one<S>(in, offset);
    kakuhen::util::serialize::deserialize_one<S>(in, size);
    kakuhen::util::serialize::deserialize_one<S>(in, n_bins);
    kakuhen::util::serialize::deserialize_one<S>(in, stride);
  }

  /**
   * @brief Compares two metadata objects for equality.
   */
  [[nodiscard]] bool operator==(const AxisMetadata& other) const noexcept {
    return type == other.type && flow == other.flow && offset == other.offset &&
           size == other.size && n_bins == other.n_bins && stride == other.stride;
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
  explicit AxisView(const metadata_type& meta) noexcept : meta_{meta} {}

  /**
   * @brief Maps a coordinate to its corresponding bin index.
   * @param axis_data Shared storage containing the axis parameters.
   * @param x The coordinate value to map.
   * @return The bin index (0 to n_bins - 1) multiplied by the stride.
   */
  [[nodiscard]] S index(const AxisData<T, S>& axis_data, const T& x) const noexcept {
    return static_cast<const Derived*>(this)->index_impl(axis_data, x) * meta_.stride;
  }

  /**
   * @brief Returns the bin edges of the regular bins.
   * @param axis_data Shared storage containing the axis parameters.
   * @return A vector containing the regular bin edges.
   */
  [[nodiscard]] std::vector<T> edges(const AxisData<T, S>& axis_data) const {
    return static_cast<const Derived*>(this)->edges_impl(axis_data);
  }

  /**
   * @brief Access the underlying metadata.
   */
  [[nodiscard]] const metadata_type& metadata() const noexcept {
    return meta_;
  }

  /**
   * @brief Get the total number of bins.
   */
  [[nodiscard]] S n_bins() const noexcept {
    return meta_.n_bins;
  }

  /**
   * @brief Get the offset into the shared AxisData storage.
   */
  [[nodiscard]] S offset() const noexcept {
    return meta_.offset;
  }

  /**
   * @brief Get the number of data points used by this axis in shared storage.
   */
  [[nodiscard]] S size() const noexcept {
    return meta_.size;
  }

  /**
   * @brief Get the stride multiplier for this axis.
   */
  [[nodiscard]] S stride() const noexcept {
    return meta_.stride;
  }

  /**
   * @brief Get the flow policy for this axis.
   */
  [[nodiscard]] UOFlowPolicy flow() const noexcept {
    return meta_.flow;
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
  metadata_type meta_;  //!< Axis location and configuration.
};

/**
 * @brief Represents an axis with uniformly spaced bins.
 *
 * Maps coordinates to indices in O(1) time using a linear transformation.
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
  explicit UniformAxis(const metadata_type& meta) noexcept : Base(meta) {
    assert(meta.type == AxisType::Uniform);
  }

  /**
   * @brief Constructs a UniformAxis and registers its parameters in AxisData.
   *
   * @param data Shared axis data storage.
   * @param n_bins Number of regular bins.
   * @param min The lower bound of the first regular bin.
   * @param max The upper bound of the last regular bin.
   * @param flow Flow policy for underflow/overflow bins.
   * @throws std::invalid_argument if parameters are invalid.
   */
  UniformAxis(AxisData<T, S>& data, S n_bins, const T& min, const T& max,
              UOFlowPolicy flow = UOFlowPolicy::Both)
      : Base(validate_and_create(data, n_bins, min, max, flow)) {}

  /**
   * @brief Maps a coordinate to a bin index using uniform mapping.
   */
  [[nodiscard]] S index_impl(const AxisData<T, S>& axis_data, const T& x) const noexcept {
    const T& min_val = axis_data[meta_.offset];
    const T& max_val = axis_data[meta_.offset + 1];
    const T& scale_val = axis_data[meta_.offset + 2];

    const bool uf = has_underflow(meta_.flow);
    const bool of = has_overflow(meta_.flow);

    if (x < min_val) return uf ? 0 : std::numeric_limits<S>::max();
    if (x >= max_val) return of ? (meta_.n_bins - 1) : std::numeric_limits<S>::max();

    return static_cast<S>((uf ? 1 : 0) + (x - min_val) * scale_val);
  }

  /**
   * @brief Implementation for retrieving uniform bin edges.
   */
  [[nodiscard]] std::vector<T> edges_impl(const AxisData<T, S>& axis_data) const {
    const T& min_val = axis_data[meta_.offset];
    const T& max_val = axis_data[meta_.offset + 1];
    const S n_reg_bins = meta_.n_bins - flow_count(meta_.flow);
    std::vector<T> res;
    res.reserve(static_cast<std::size_t>(n_reg_bins + 1));
    const T step = (max_val - min_val) / static_cast<T>(n_reg_bins);
    for (S i = 0; i <= n_reg_bins; ++i) {
      res.push_back(min_val + static_cast<T>(i) * step);
    }
    // Ensure the last edge is exactly max_val
    if (!res.empty()) res.back() = max_val;
    return res;
  }

 private:
  /**
   * @brief Validates parameters and appends data to storage.
   */
  static metadata_type validate_and_create(AxisData<T, S>& data, S n_bins, const T& min,
                                           const T& max, UOFlowPolicy flow) {
    if (n_bins == 0) throw std::invalid_argument("UniformAxis: n_bins must be > 0");
    if (min >= max) throw std::invalid_argument("UniformAxis: min must be < max");
    return {AxisType::Uniform,
            flow,
            data.add_data(min, max, static_cast<T>(n_bins) / (max - min)),
            S(3),
            static_cast<S>(n_bins + flow_count(flow)),
            1};
  }
};

/**
 * @brief Represents an axis with variable bin widths.
 *
 * Maps coordinates to indices in O(log N) time using binary search over edges.
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
  explicit VariableAxis(const metadata_type& meta) noexcept : Base(meta) {
    assert(meta.type == AxisType::Variable);
  }

  /**
   * @brief Constructs a VariableAxis and registers its edges in AxisData.
   *
   * @param data The shared axis data storage.
   * @param edges The bin edges (must be sorted).
   * @param flow Flow policy for underflow/overflow bins.
   * @throws std::invalid_argument if parameters are invalid.
   */
  VariableAxis(AxisData<T, S>& data, const std::vector<T>& edges,
               UOFlowPolicy flow = UOFlowPolicy::Both)
      : Base(validate_and_create(data, edges, flow)) {}

  /**
   * @brief Maps a coordinate to a bin index using binary search.
   */
  [[nodiscard]] S index_impl(const AxisData<T, S>& axis_data, const T& x) const noexcept {
    auto begin = axis_data.data().begin() + meta_.offset;
    auto end = begin + meta_.size;

    const bool uf = has_underflow(meta_.flow);
    const bool of = has_overflow(meta_.flow);

    if (x < *begin) return uf ? 0 : std::numeric_limits<S>::max();
    if (x >= *(end - 1)) return of ? (meta_.n_bins - 1) : std::numeric_limits<S>::max();

    auto it = std::upper_bound(begin, end, x);
    const S local_idx = static_cast<S>(std::distance(begin, it)) - 1;
    return (uf ? 1 : 0) + local_idx;
  }

  /**
   * @brief Implementation for retrieving variable bin edges.
   */
  [[nodiscard]] std::vector<T> edges_impl(const AxisData<T, S>& axis_data) const {
    auto begin = axis_data.data().begin() + meta_.offset;
    auto end = begin + meta_.size;
    return std::vector<T>(begin, end);
  }

 private:
  /**
   * @brief Validates edges and appends to storage.
   */
  static metadata_type validate_and_create(AxisData<T, S>& data, const std::vector<T>& edges,
                                           UOFlowPolicy flow) {
    if (edges.size() < 2) throw std::invalid_argument("VariableAxis: requires at least 2 edges");
    if (!std::is_sorted(edges.begin(), edges.end())) {
      throw std::invalid_argument("VariableAxis: edges must be sorted");
    }
    return {AxisType::Variable,
            flow,
            data.add_data(edges),
            static_cast<S>(edges.size()),
            static_cast<S>(edges.size() - 1 + flow_count(flow)),
            1};
  }
};

/**
 * @brief Variant type holding any supported axis implementation.
 */
template <typename T, typename S>
using AxisVariant = std::variant<std::monostate, UniformAxis<T, S>, VariableAxis<T, S>>;

/**
 * @brief Factory function to restore an axis view from its metadata.
 *
 * @tparam T The coordinate value type.
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
