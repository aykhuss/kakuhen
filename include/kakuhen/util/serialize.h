#pragma once

#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace kakuhen::util::serialize {

/// @name Core serialization routines
/// @{

/*!
 * @brief Writes a specified number of bytes to an output stream.
 *
 * @param out The output stream to write to.
 * @param data A pointer to the data buffer.
 * @param size The number of bytes to write.
 */
inline void write_bytes(std::ostream& out, const void* data, size_t size) {
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

/*!
 * @brief Reads a specified number of bytes from an input stream.
 *
 * @param in The input stream to read from.
 * @param data A pointer to the buffer where the read bytes will be stored.
 * @param size The number of bytes to read.
 * @throws std::runtime_error if the expected number of bytes cannot be read.
 */
inline void read_bytes(std::istream& in, void* data, size_t size) {
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
  if (in.gcount() != static_cast<std::streamsize>(size)) {
    throw std::runtime_error("Failed to read expected number of bytes");
  }
}

/// @}

/// @name SFINAE detection idioms
/// @{

/*!
 * @brief SFINAE helper to detect if a type has a `serialize` method.
 *
 * This struct checks if a type `T` provides a `void serialize(std::ostream&) const`
 * method. It is used to conditionally enable custom serialization logic.
 *
 * @tparam T The type to check.
 * @tparam Enable Internal detail to enable/disable the specialization.
 */
template <typename, typename = void>
struct has_serialize : std::false_type {};

template <typename T>
struct has_serialize<
    T, std::void_t<decltype(std::declval<const T&>().serialize(std::declval<std::ostream&>()))>>
    : std::true_type {};

/*!
 * @brief SFINAE helper to detect if a type has a `deserialize` method.
 *
 * This struct checks if a type `T` provides a `void deserialize(std::istream&)`
 * method. It is used to conditionally enable custom deserialization logic.
 *
 * @tparam T The type to check.
 * @tparam Enable Internal detail to enable/disable the specialization.
 */
template <typename, typename = void>
struct has_deserialize : std::false_type {};

template <typename T>
struct has_deserialize<
    T, std::void_t<decltype(std::declval<T&>().deserialize(std::declval<std::istream&>()))>>
    : std::true_type {};

/// @}

/// @name Generic serialization
/// @{

/*!
 * @brief Serializes a single object to an output stream.
 *
 * This function handles serialization for both types with custom `serialize()`
 * methods and trivially copyable (POD) types. For types with a custom `serialize()`
 * method, it calls that method. For POD types, it performs a byte-wise write.
 *
 * @tparam T The type of the object to serialize.
 * @param out The output stream to write to.
 * @param obj The object to serialize.
 */
template <typename T>
void serialize_one(std::ostream& out, const T& obj) {
  if constexpr (has_serialize<T>::value) {
    obj.serialize(out);
  } else {
    static_assert(std::is_trivially_copyable_v<T>, "Only POD types allowed");
    write_bytes(out, &obj, sizeof(T));
  }
}

/*!
 * @brief Deserializes a single object from an input stream.
 *
 * This function handles deserialization for both types with custom `deserialize()`
 * methods and trivially copyable (POD) types. For types with a custom `deserialize()`
 * method, it calls that method. For POD types, it performs a byte-wise read.
 *
 * @tparam T The type of the object to deserialize.
 * @param in The input stream to read from.
 * @param obj The object to deserialize into.
 */
template <typename T>
void deserialize_one(std::istream& in, T& obj) {
  if constexpr (has_deserialize<T>::value) {
    obj.deserialize(in);
  } else {
    static_assert(std::is_trivially_copyable_v<T>, "Only POD types allowed");
    read_bytes(in, &obj, sizeof(T));
  }
}

/// @}

/// @name Array and Range serialization
/// @{

/*!
 * @brief Serializes a C-style array of objects to an output stream.
 *
 * This function iterates through a raw pointer array and serializes each
 * element using `serialize_one`.
 *
 * @tparam T The type of elements in the array.
 * @param out The output stream to write to.
 * @param ptr A pointer to the first element of the array.
 * @param count The number of elements in the array.
 */
template <typename T>
void serialize_array(std::ostream& out, const T* ptr, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    serialize_one<T>(out, ptr[i]);
  }
}

/*!
 * @brief Deserializes a C-style array of objects from an input stream.
 *
 * This function iterates through a raw pointer array and deserializes each
 * element using `deserialize_one`.
 *
 * @tparam T The type of elements in the array.
 * @param in The input stream to read from.
 * @param ptr A pointer to the buffer where the deserialized elements will be stored.
 * @param count The number of elements to deserialize.
 */
template <typename T>
void deserialize_array(std::istream& in, T* ptr, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    deserialize_one<T>(in, ptr[i]);
  }
}

/*!
 * @brief Serializes a range of objects defined by iterators.
 *
 * Each element in the range `[first, last)` is serialized using `serialize_one`.
 *
 * @tparam Iterator The iterator type defining the range.
 * @param out The output stream to write to.
 * @param first An iterator to the beginning of the range.
 * @param last An iterator to the end of the range.
 */
template <typename Iterator>
void serialize_range(std::ostream& out, Iterator first, Iterator last) {
  for (; first != last; ++first) {
    serialize_one(out, *first);
  }
}

/*!
 * @brief Serializes the contents of a container.
 *
 * This function uses `serialize_range` to serialize all elements in a container
 * that provides `std::begin` and `std::end`.
 *
 * @tparam Container The container type.
 * @param out The output stream to write to.
 * @param container The container to serialize.
 */
template <typename Container>
void serialize_container(std::ostream& out, const Container& container) {
  serialize_range(out, std::begin(container), std::end(container));
}

/*!
 * @brief Deserializes a range of objects defined by iterators.
 *
 * Each element in the range `[first, last)` is deserialized using `deserialize_one`.
 *
 * @tparam Iterator The iterator type defining the range.
 * @param in The input stream to read from.
 * @param first An iterator to the beginning of the range (where deserialized objects will be stored).
 * @param last An iterator to the end of the range.
 */
template <typename Iterator>
void deserialize_range(std::istream& in, Iterator first, Iterator last) {
  for (; first != last; ++first) {
    deserialize_one(in, *first);
  }
}

/*!
 * @brief Deserializes into the contents of a container.
 *
 * This function uses `deserialize_range` to deserialize elements into a container
 * that provides `std::begin` and `std::end`.
 *
 * @tparam Container The container type.
 * @param in The input stream to read from.
 * @param container The container to deserialize into.
 */
template <typename Container>
void deserialize_container(std::istream& in, Container& container) {
  deserialize_range(in, std::begin(container), std::end(container));
}

/// @}

}  // namespace kakuhen::util::serialize