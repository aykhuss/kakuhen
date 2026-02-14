#pragma once

#include <concepts>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <type_traits>

namespace kakuhen::integrator {

/*!
 * @brief Defines feature flags for integrators.
 *
 * These flags are used to indicate which features a particular integrator
 * supports, such as saving/restoring state, accumulating data, or adaptive
 * integration.
 */
enum class IntegratorFeature : std::uint16_t {
  NONE = 0,         //!< No special features.
  STATE = 1u << 0,  //!< Supports saving and restoring its internal state.
  DATA = 1u << 1,   //!< Supports accumulating and dumping intermediate data.
  ADAPT = 1u << 2,  //!< Supports adaptive integration (e.g., grid refinement).
};  // enum class IntegratorFeature

/*!
 * @brief Converts an enum class to its underlying integer type.
 *
 * This acts as a backport of `std::to_underlying` (C++23) for earlier standards.
 *
 * @tparam Enum The enum class type.
 * @param e The enum value.
 * @return The underlying integer value.
 */
template <typename Enum>
  requires std::is_enum_v<Enum>
constexpr auto to_underlying(Enum e) noexcept {
  return static_cast<std::underlying_type_t<Enum>>(e);
}

/// @name Bitwise Operators
/// @{

/*!
 * @brief Bitwise OR operator for IntegratorFeature.
 */
constexpr IntegratorFeature operator|(IntegratorFeature lhs, IntegratorFeature rhs) noexcept {
  return static_cast<IntegratorFeature>(to_underlying(lhs) | to_underlying(rhs));
}
/*!
 * @brief Bitwise AND operator for IntegratorFeature.
 */
constexpr IntegratorFeature operator&(IntegratorFeature lhs, IntegratorFeature rhs) noexcept {
  return static_cast<IntegratorFeature>(to_underlying(lhs) & to_underlying(rhs));
}
/*!
 * @brief Bitwise XOR operator for IntegratorFeature.
 */
constexpr IntegratorFeature operator^(IntegratorFeature lhs, IntegratorFeature rhs) noexcept {
  return static_cast<IntegratorFeature>(to_underlying(lhs) ^ to_underlying(rhs));
}
/*!
 * @brief Bitwise NOT operator for IntegratorFeature.
 */
constexpr IntegratorFeature operator~(IntegratorFeature f) noexcept {
  return static_cast<IntegratorFeature>(~to_underlying(f));
}

/*!
 * @brief Compound bitwise OR assignment operator for IntegratorFeature.
 */
inline IntegratorFeature& operator|=(IntegratorFeature& lhs, IntegratorFeature rhs) noexcept {
  return lhs = lhs | rhs;
}
/*!
 * @brief Compound bitwise AND assignment operator for IntegratorFeature.
 */
inline IntegratorFeature& operator&=(IntegratorFeature& lhs, IntegratorFeature rhs) noexcept {
  return lhs = lhs & rhs;
}
/*!
 * @brief Compound bitwise XOR assignment operator for IntegratorFeature.
 */
inline IntegratorFeature& operator^=(IntegratorFeature& lhs, IntegratorFeature rhs) noexcept {
  return lhs = lhs ^ rhs;
}

/// @}

namespace detail {

/*!
 * @brief Checks if a specific feature flag is set.
 *
 * @param value The feature set to check.
 * @param flag The specific flag to look for.
 * @return True if the flag is set, false otherwise.
 */
constexpr bool has_flag(IntegratorFeature value, IntegratorFeature flag) noexcept {
  return to_underlying(value & flag) != 0;
}

/*!
 * @brief Concept for integrators that implement adaptation.
 *
 * Requires a callable `adapt()` method with `void` return type.
 */
template <typename T>
concept HasAdapt = requires(T t) {
  { t.adapt() } -> std::same_as<void>;
};

/*!
 * @brief Concept for integrators that support state serialization.
 *
 * Requires `write_state_stream(std::ostream&)` and
 * `read_state_stream(std::istream&)`, both returning `void`.
 */
template <typename T>
concept HasStateStream = requires(T t, std::ostream& out, std::istream& in) {
  { t.write_state_stream(out) } -> std::same_as<void>;
  { t.read_state_stream(in) } -> std::same_as<void>;
};

/*!
 * @brief Concept for integrators that support data serialization/accumulation.
 *
 * Requires `write_data_stream(std::ostream&)`,
 * `read_data_stream(std::istream&)`, and
 * `accumulate_data_stream(std::istream&)`, all returning `void`.
 */
template <typename T>
concept HasDataStream = requires(T t, std::ostream& out, std::istream& in) {
  { t.write_data_stream(out) } -> std::same_as<void>;
  { t.read_data_stream(in) } -> std::same_as<void>;
  { t.accumulate_data_stream(in) } -> std::same_as<void>;
};

/*!
 * @brief Concept for integrators that provide path prefix generation.
 *
 * Requires a callable `prefix(bool)` that returns `std::string`.
 */
template <typename T>
concept HasPrefix = requires(T t, bool with_hash) {
  { t.prefix(with_hash) } -> std::same_as<std::string>;
};

/// @}

}  // namespace detail

}  // namespace kakuhen::integrator
