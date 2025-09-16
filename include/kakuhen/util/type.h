#pragma once

#include <cstdint>

namespace kakuhen::util::type {

enum class TypeId : uint8_t {
  UNKNOWN,
  BOOL,
  INT8,
  INT16,
  INT32,
  INT64,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  FLOAT,
  DOUBLE
};

//> default: cannot recognize
template <typename T>
constexpr TypeId get_type_id() {
  return TypeId::UNKNOWN;
}
//> specializations
template <>
constexpr TypeId get_type_id<bool>() {
  return TypeId::BOOL;
}
template <>
constexpr TypeId get_type_id<int8_t>() {
  return TypeId::INT8;
}
template <>
constexpr TypeId get_type_id<int16_t>() {
  return TypeId::INT16;
}
template <>
constexpr TypeId get_type_id<int32_t>() {
  return TypeId::INT32;
}
template <>
constexpr TypeId get_type_id<int64_t>() {
  return TypeId::INT64;
}
template <>
constexpr TypeId get_type_id<uint8_t>() {
  return TypeId::UINT8;
}
template <>
constexpr TypeId get_type_id<uint16_t>() {
  return TypeId::UINT16;
}
template <>
constexpr TypeId get_type_id<uint32_t>() {
  return TypeId::UINT32;
}
template <>
constexpr TypeId get_type_id<uint64_t>() {
  return TypeId::UINT64;
}
template <>
constexpr TypeId get_type_id<float>() {
  return TypeId::FLOAT;
}
template <>
constexpr TypeId get_type_id<double>() {
  return TypeId::DOUBLE;
}

//> In case I don't understand the type, use the size of the type (as a negative
// value) > to get some compatibility information for serialization

template <typename T>
constexpr int16_t get_type_or_size() {
  constexpr TypeId type_id = get_type_id<T>();
  if constexpr (type_id != TypeId::UNKNOWN) {
    return static_cast<int16_t>(type_id);
  } else {
    return -static_cast<int16_t>(sizeof(T));
  }
}

}  // namespace kakuhen::util::type
