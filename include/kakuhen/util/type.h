#pragma once

#include <cstdint>
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

///------------------------------------
/// the enum class with all id's
///------------------------------------
enum class TypeId : uint8_t {
  UNKNOWN = 0,
#define DEFINE_ENUM_ENTRY(_, NAME) NAME,
  KAKUHEN_TYPE_LIST(DEFINE_ENUM_ENTRY)
#undef DEFINE_ENUM_ENTRY
};

///------------------------------------
/// type -> TypeId
///------------------------------------

/// //> default: cannot recognize
/// template <typename T>
/// constexpr TypeId get_type_id() {
///   return TypeId::UNKNOWN;
/// }
///
/// //> specializations
/// #define DEFINE_TYPE_TO_ID(TYPE, NAME)    \
///   template <>                            \
///   constexpr TypeId get_type_id<TYPE>() { \
///     return TypeId::NAME;                 \
///   }
/// KAKUHEN_TYPE_LIST(DEFINE_TYPE_TO_ID)
/// #undef DEFINE_TYPE_TO_ID

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

//> In case I don't recognize the type, use the size of the type (negative)
//> to get some compatibility information for serialization
template <typename T>
constexpr int16_t get_type_or_size() {
  constexpr TypeId type_id = get_type_id<T>();
  if constexpr (type_id != TypeId::UNKNOWN) {
    return static_cast<int16_t>(type_id);
  } else {
    return -static_cast<int16_t>(sizeof(T));
  }
}

///------------------------------------
/// TypeId -> type
///------------------------------------

//> default: cannot recognize
template <TypeId>
struct TypeFromId {
  using type = void;
};

//> specializations
#define DEFINE_ID_TO_TYPE(TYPE, NAME) \
  template <>                         \
  struct TypeFromId<TypeId::NAME> {   \
    using type = TYPE;                \
  };
KAKUHEN_TYPE_LIST(DEFINE_ID_TO_TYPE)
#undef DEFINE_ID_TO_TYPE

//> shorthand
template <TypeId ID>
using type_from_id_t = typename TypeFromId<ID>::type;

}  // namespace kakuhen::util::type

#undef KAKUHEN_TYPE_LIST
