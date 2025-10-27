#pragma once

#include <cstdint>
#include <iostream>
#include <type_traits>

namespace kakuhen::integrator {

//> Integrator feature flags
enum class IntegratorFeature : std::uint16_t {
  NONE = 0,
  STATE = 1u << 0,  // supports saving/restoring state
  DATA = 1u << 1,   // supports data accumulation with file dump
  ADAPT = 1u << 2,  // supports adaptive integration
};  // enum class IntegratorFeature

// backport of std::to_underlying for C++20 and earlier
template <typename Enum>
constexpr auto to_underlying(Enum e) noexcept {
  return static_cast<std::underlying_type_t<Enum>>(e);
}

//> Bitwise operators
constexpr IntegratorFeature operator|(IntegratorFeature lhs, IntegratorFeature rhs) noexcept {
  return static_cast<IntegratorFeature>(to_underlying(lhs) | to_underlying(rhs));
}
constexpr IntegratorFeature operator&(IntegratorFeature lhs, IntegratorFeature rhs) noexcept {
  return static_cast<IntegratorFeature>(to_underlying(lhs) & to_underlying(rhs));
}
constexpr IntegratorFeature operator^(IntegratorFeature lhs, IntegratorFeature rhs) noexcept {
  return static_cast<IntegratorFeature>(to_underlying(lhs) ^ to_underlying(rhs));
}
constexpr IntegratorFeature operator~(IntegratorFeature f) noexcept {
  return static_cast<IntegratorFeature>(~to_underlying(f));
}

//> Compound assignment
inline IntegratorFeature& operator|=(IntegratorFeature& lhs, IntegratorFeature rhs) noexcept {
  return lhs = lhs | rhs;
}
inline IntegratorFeature& operator&=(IntegratorFeature& lhs, IntegratorFeature rhs) noexcept {
  return lhs = lhs & rhs;
}
inline IntegratorFeature& operator^=(IntegratorFeature& lhs, IntegratorFeature rhs) noexcept {
  return lhs = lhs ^ rhs;
}

namespace detail {

//> Flag test helper
constexpr bool has_flag(IntegratorFeature value, IntegratorFeature flag) noexcept {
  return to_underlying(value & flag) != 0;
}

/// concepts were replaced by
// template <typename T>
// concept HasAdapt = requires(T t) {
//   { t.adapt() } -> std::same_as<void>;
// };
// template <typename T>
// concept HasStateStream = requires(T t, std::ostream& out, std::istream& in) {
//   { t.write_state_stream(out) } -> std::same_as<void>;
//   { t.read_state_stream(in) } -> std::same_as<void>;
// };
// template <typename T>
// concept HasDataStream = requires(T t, std::ostream& out, std::istream& in) {
//   { t.write_data_stream(out) } -> std::same_as<void>;
//   { t.read_data_stream(in) } -> std::same_as<void>;
//   { t.accumulate_data_stream(in) } -> std::same_as<void>;
// };
// template <typename T>
// concept HasPrefix = requires(T t, bool b) {
//   { t.prefix(b) } -> std::same_as<std::string>;
// };

// SFINAE detection idiom (C++17-friendly)
template <typename, typename = void>
struct has_adapt : std::false_type {};

template <typename T>
struct has_adapt<T, std::void_t<decltype(std::declval<T>().adapt())>> : std::true_type {};

template <typename, typename = void>
struct has_state_stream : std::false_type {};

template <typename T>
struct has_state_stream<
    T, std::void_t<decltype(std::declval<T>().write_state_stream(std::declval<std::ostream&>())),
                   decltype(std::declval<T>().read_state_stream(std::declval<std::istream&>()))>>
    : std::true_type {};

template <typename, typename = void>
struct has_data_stream : std::false_type {};

template <typename T>
struct has_data_stream<
    T,
    std::void_t<decltype(std::declval<T>().write_data_stream(std::declval<std::ostream&>())),
                decltype(std::declval<T>().read_data_stream(std::declval<std::istream&>())),
                decltype(std::declval<T>().accumulate_data_stream(std::declval<std::istream&>()))>>
    : std::true_type {};

template <typename, typename = void>
struct has_prefix : std::false_type {};

template <typename T>
struct has_prefix<T, std::void_t<decltype(std::declval<T>().prefix(true))>> : std::true_type {};

}  // namespace detail

}  // namespace kakuhen::integrator
