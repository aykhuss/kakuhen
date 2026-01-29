#pragma once

#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace kakuhen::util::serialize {

/// @name SFINAE detection idioms
/// @{

/**
 * @brief SFINAE helper to detect if a type has a `serialize` method.
 */
template <typename, typename = void>
struct has_serialize : std::false_type {};

template <typename T>
struct has_serialize<
    T, std::void_t<decltype(std::declval<const T&>().serialize(std::declval<std::ostream&>()))>>
    : std::true_type {};

/**
 * @brief SFINAE helper to detect if a type has a `deserialize` method.
 */
template <typename, typename = void>
struct has_deserialize : std::false_type {};

template <typename T>
struct has_deserialize<
    T, std::void_t<decltype(std::declval<T&>().deserialize(std::declval<std::istream&>()))>>
    : std::true_type {};

/// @}

/// @name Core serialization routines
/// @{

/**
 * @brief Writes a specified number of bytes to an output stream.
 *
 * @param out The output stream to write to.
 * @param data A pointer to the data buffer.
 * @param size The number of bytes to write.
 */
inline void write_bytes(std::ostream& out, const void* data, std::size_t size) {
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

/**
 * @brief Reads a specified number of bytes from an input stream.
 *
 * @param in The input stream to read from.
 * @param data A pointer to the buffer where the read bytes will be stored.
 * @param size The number of bytes to read.
 * @throws std::runtime_error if the expected number of bytes cannot be read.
 */
inline void read_bytes(std::istream& in, void* data, std::size_t size) {
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
  if (static_cast<std::size_t>(in.gcount()) != size) {
    throw std::runtime_error("Failed to read expected number of bytes from stream.");
  }
}

/// @}

/// @name Size serialization
/// @{

/**
 * @brief Serializes a size value using a stable 64-bit unsigned integer format.
 */
inline void serialize_size(std::ostream& out, std::size_t size) {
  const auto s = static_cast<std::uint64_t>(size);
  write_bytes(out, &s, sizeof(s));
}

/**
 * @brief Deserializes a size value from a 64-bit unsigned integer format.
 */
inline void deserialize_size(std::istream& in, std::size_t& size) {
  std::uint64_t s;
  read_bytes(in, &s, sizeof(s));
  size = static_cast<std::size_t>(s);
}

/// @}

/// @name Generic serialization
/// @{

/**
 * @brief Serializes a single object to an output stream.
 *
 * Dispatches to:
 * 1. Special handling for `std::string`.
 * 2. `obj.serialize(out)` if the type satisfies `has_serialize`.
 * 3. Byte-wise copy for trivially copyable non-pointer types.
 *
 * @tparam T The type of the object to serialize.
 * @param out The output stream to write to.
 * @param obj The object to serialize.
 */
template <typename T>
void serialize_one(std::ostream& out, const T& obj) {
  using DT = std::decay_t<T>;

  if constexpr (std::is_same_v<DT, std::string>) {
    serialize_size(out, obj.size());
    if (!obj.empty()) {
      write_bytes(out, obj.data(), obj.size());
    }
  } else if constexpr (has_serialize<DT>::value) {
    obj.serialize(out);
  } else {
    static_assert(std::is_trivially_copyable_v<DT>,
                  "Type must be trivially copyable, std::string, or provide serialize().");
    static_assert(!std::is_pointer_v<DT>, "Serializing raw pointers by value is not allowed.");
    write_bytes(out, &obj, sizeof(DT));
  }
}

/**
 * @brief Deserializes a single object from an input stream.
 *
 * Dispatches to:
 * 1. Special handling for `std::string`.
 * 2. `obj.deserialize(in)` if the type satisfies `has_deserialize`.
 * 3. Byte-wise copy for trivially copyable non-pointer types.
 *
 * @tparam T The type of the object to deserialize.
 * @param in The input stream to read from.
 * @param obj The object to deserialize into.
 */
template <typename T>
void deserialize_one(std::istream& in, T& obj) {
  using DT = std::decay_t<T>;

  if constexpr (std::is_same_v<DT, std::string>) {
    std::size_t size;
    deserialize_size(in, size);
    obj.resize(size);
    if (size > 0) {
      read_bytes(in, obj.data(), size);
    }
  } else if constexpr (has_deserialize<DT>::value) {
    obj.deserialize(in);
  } else {
    static_assert(std::is_trivially_copyable_v<DT>,
                  "Type must be trivially copyable, std::string, or provide deserialize().");
    static_assert(!std::is_pointer_v<DT>, "Deserializing raw pointers by value is not allowed.");
    read_bytes(in, &obj, sizeof(DT));
  }
}

/// @}

/// @name Array and Range serialization
/// @{

/**
 * @brief Serializes a C-style array of objects.
 *
 * Uses optimized byte-wise write if the element type is trivially copyable.
 *
 * @tparam T The type of elements in the array.
 * @param out The output stream.
 * @param ptr Pointer to the start of the array.
 * @param count Number of elements.
 */
template <typename T>
void serialize_array(std::ostream& out, const T* ptr, std::size_t count) {
  if (count == 0) return;
  if constexpr (std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>) {
    write_bytes(out, ptr, count * sizeof(T));
  } else {
    for (std::size_t i = 0; i < count; ++i) {
      serialize_one(out, ptr[i]);
    }
  }
}

/**
 * @brief Deserializes a C-style array of objects.
 *
 * Uses optimized byte-wise read if the element type is trivially copyable.
 */
template <typename T>
void deserialize_array(std::istream& in, T* ptr, std::size_t count) {
  if (count == 0) return;
  if constexpr (std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>) {
    read_bytes(in, ptr, count * sizeof(T));
  } else {
    for (std::size_t i = 0; i < count; ++i) {
      deserialize_one(in, ptr[i]);
    }
  }
}

/**
 * @brief Serializes a range of objects defined by iterators.
 */
template <typename Iterator>
void serialize_range(std::ostream& out, Iterator first, Iterator last) {
  for (; first != last; ++first) {
    serialize_one(out, *first);
  }
}

/**
 * @brief Deserializes a range of objects defined by iterators.
 */
template <typename Iterator>
void deserialize_range(std::istream& in, Iterator first, Iterator last) {
  for (; first != last; ++first) {
    deserialize_one(in, *first);
  }
}

/**
 * @brief Serializes the contents of a container.
 *
 * Provides optimized paths for contiguous containers of POD types (e.g. `std::vector<double>`).
 * Special handling ensures `std::vector<bool>` is treated correctly via iteration.
 */
template <typename Container>
void serialize_container(std::ostream& out, const Container& container) {
  using T = typename Container::value_type;
  if constexpr (std::is_trivially_copyable_v<T> && !std::is_same_v<T, bool> &&
                !std::is_pointer_v<T>) {
    if constexpr (requires(const Container& c) {
                    c.data();
                    c.size();
                  }) {
      serialize_array(out, container.data(), container.size());
    } else {
      serialize_range(out, std::begin(container), std::end(container));
    }
  } else {
    serialize_range(out, std::begin(container), std::end(container));
  }
}

/**
 * @brief Deserializes the contents into an existing container.
 *
 * The container must have the correct size allocated before calling this function.
 */
template <typename Container>
void deserialize_container(std::istream& in, Container& container) {
  using T = typename Container::value_type;
  if constexpr (std::is_trivially_copyable_v<T> && !std::is_same_v<T, bool> &&
                !std::is_pointer_v<T>) {
    if constexpr (requires(Container& c) {
                    c.data();
                    c.size();
                  }) {
      deserialize_array(in, container.data(), container.size());
    } else {
      deserialize_range(in, std::begin(container), std::end(container));
    }
  } else {
    deserialize_range(in, std::begin(container), std::end(container));
  }
}

/// @}

}  // namespace kakuhen::util::serialize
