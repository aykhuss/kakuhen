#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

#define KAKUHEN_TYPE_LIST(X) \
  X(bool, BOOL)              \
  X(int8_t, INT8)            \
  X(int16_t, INT16)          \
  X(int32_t, INT32)          \
  X(int64_t, INT64)          \
  X(uint8_t, UINT8)          \
  X(uint16_t, UINT16)        \
  X(uint32_t, UINT32)        \
  X(uint64_t, UINT64)        \
  X(float, FLOAT)            \
  X(double, DOUBLE)

namespace kakuhen::util::type {

/// @name Type Identification
/// @{

/*!
 * @brief Enum class defining identifiers for various supported types.
 *
 * This enum lists all the types supported by the type identification system.
 * The `UNKNOWN` value is used for types that are not explicitly supported.
 */
enum class TypeId : uint8_t {
  UNKNOWN = 0,
#define DEFINE_ENUM_ENTRY(_, NAME) NAME,
  KAKUHEN_TYPE_LIST(DEFINE_ENUM_ENTRY)
#undef DEFINE_ENUM_ENTRY
};

/*!
 * @brief Gets the `TypeId` corresponding to a given type `T`.
 *
 * This function maps a C++ type to its corresponding `TypeId` enum value.
 * It supports standard arithmetic types and their fixed-width integer equivalents.
 *
 * @tparam T The type to identify.
 * @return The `TypeId` corresponding to `T`, or `TypeId::UNKNOWN` if not found.
 */
template <typename T>
constexpr TypeId get_type_id() {
#define DEFINE_TYPE_TO_ID(TYPE, NAME) \
  if constexpr (std::is_same_v<T, TYPE>) return TypeId::NAME;
  KAKUHEN_TYPE_LIST(DEFINE_TYPE_TO_ID)
#undef DEFINE_TYPE_TO_ID

  // map native signed integers by size
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    if constexpr (sizeof(T) == 1) return TypeId::INT8;
    if constexpr (sizeof(T) == 2) return TypeId::INT16;
    if constexpr (sizeof(T) == 4) return TypeId::INT32;
    if constexpr (sizeof(T) == 8) return TypeId::INT64;
  }

  // map native unsigned integers by size
  if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
    if constexpr (sizeof(T) == 1) return TypeId::UINT8;
    if constexpr (sizeof(T) == 2) return TypeId::UINT16;
    if constexpr (sizeof(T) == 4) return TypeId::UINT32;
    if constexpr (sizeof(T) == 8) return TypeId::UINT64;
  }

  return TypeId::UNKNOWN;
}

/*!
 * @brief Gets the `TypeId` as a signed 16-bit integer, or the negative size of the type.
 *
 * If the type is recognized, its `TypeId` is returned as a positive integer.
 * If the type is unknown, its size (in bytes) is returned as a negative integer.
 * This is useful for basic compatibility checking during serialization.
 *
 * @note This function returns an `int16_t`. If the type is unknown and its size
 * exceeds 32767 bytes (32 KB), the result will overflow/wrap, potentially leading
 * to incorrect compatibility checks.
 *
 * @tparam T The type to query.
 * @return The `TypeId` or negative size of `T`.
 */
template <typename T>
constexpr int16_t get_type_or_size() {
  constexpr TypeId type_id = get_type_id<T>();
  if constexpr (type_id != TypeId::UNKNOWN) {
    return static_cast<int16_t>(type_id);
  } else {
    return -static_cast<int16_t>(sizeof(T));
  }
}

/// @}

/// @name Type Mapping
/// @{

/*!
 * @brief Helper struct to map a `TypeId` back to a C++ type.
 *
 * The `type` member alias will be `void` if the ID is unknown or `TypeId::UNKNOWN`.
 * Specializations are generated for each supported type.
 *
 * @tparam ID The `TypeId` to map.
 */
template <TypeId>
struct TypeFromId {
  using type = void;
};

// specializations
#define DEFINE_ID_TO_TYPE(TYPE, NAME) \
  template <>                         \
  struct TypeFromId<TypeId::NAME> {   \
    using type = TYPE;                \
  };
KAKUHEN_TYPE_LIST(DEFINE_ID_TO_TYPE)
#undef DEFINE_ID_TO_TYPE

/*!
 * @brief Alias helper for `TypeFromId<ID>::type`.
 *
 * @tparam ID The `TypeId` to map.
 */
template <TypeId ID>
using type_from_id_t = typename TypeFromId<ID>::type;

/// @}

/// @name String Conversion
/// @{

/*!
 * @brief Converts a `TypeId` to its string representation.
 *
 * @param id The `TypeId` to convert.
 * @return A string view of the type name (e.g., "INT32"), or "UNKNOWN".
 */
constexpr std::string_view to_string(TypeId id) {
  switch (id) {
#define CASE_TYPE_TO_STRING(_, NAME) \
  case TypeId::NAME:                 \
    return #NAME;
    KAKUHEN_TYPE_LIST(CASE_TYPE_TO_STRING)
#undef CASE_TYPE_TO_STRING
    default:
      return "UNKNOWN";
  }
}

/*!
 * @brief Gets the string name of a type `T`.
 *
 * This function is a convenience wrapper around `get_type_id` and `to_string`.
 *
 * @tparam T The type to query.
 * @return A string view of the type name.
 */
template <typename T>
constexpr std::string_view get_type_name() {
  constexpr TypeId id = get_type_id<T>();
  return to_string(id);
}

/// @}

}  // namespace kakuhen::util::type

#undef KAKUHEN_TYPE_LIST
