#pragma once

#include <concepts>
#include <iosfwd>
#include <string>

namespace kakuhen::integrator {

namespace detail {

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
