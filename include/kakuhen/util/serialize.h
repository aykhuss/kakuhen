#pragma once

#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace kakuhen::util::serialize {

//--- core serialization routines

inline void write_bytes(std::ostream& out, const void* data, size_t size) {
  out.write(reinterpret_cast<const char*>(data), size);
}

inline void read_bytes(std::istream& in, void* data, size_t size) {
  in.read(reinterpret_cast<char*>(data), size);
  if (in.gcount() != static_cast<std::streamsize>(size)) {
    throw std::runtime_error("Failed to read expected number of bytes");
  }
}

//--- SFINAE detection idioms

// Check for `void serialize(std::ostream&) const`
template <typename, typename = void>
struct has_serialize : std::false_type {};

template <typename T>
struct has_serialize<
    T, std::void_t<decltype(std::declval<const T&>().serialize(std::declval<std::ostream&>()))>>
    : std::true_type {};

// Check for `void deserialize(std::istream&)`
template <typename, typename = void>
struct has_deserialize : std::false_type {};

template <typename T>
struct has_deserialize<
    T, std::void_t<decltype(std::declval<T&>().deserialize(std::declval<std::istream&>()))>>
    : std::true_type {};

//--- Generic serialization of an object

template <typename T>
void serialize_one(std::ostream& out, const T& obj) {
  if constexpr (has_serialize<T>::value) {
    // std::cout << "Using custom serialize()\n";
    obj.serialize(out);
  } else {
    // std::cout << "Using default serialize_one()\n";
    static_assert(std::is_trivially_copyable_v<T>, "Only POD types allowed");
    write_bytes(out, &obj, sizeof(T));
  }
}

template <typename T>
void deserialize_one(std::istream& in, T& obj) {
  if constexpr (has_deserialize<T>::value) {
    // std::cout << "Using custom deserialize()\n";
    obj.deserialize(in);
  } else {
    // std::cout << "Using default deserialize_one()\n";
    static_assert(std::is_trivially_copyable_v<T>, "Only POD types allowed");
    read_bytes(in, &obj, sizeof(T));
  }
}

//--- raw array pointers

template <typename T>
void serialize_array(std::ostream& out, const T* ptr, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    serialize_one<T>(out, ptr[i]);
  }
}

template <typename T>
void deserialize_array(std::istream& in, T* ptr, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    deserialize_one<T>(in, ptr[i]);
  }
}

//--- deal with containers

// Serialize a range [first, last) of objects using iterators
template <typename Iterator>
void serialize_range(std::ostream& out, Iterator first, Iterator last) {
  for (; first != last; ++first) {
    serialize_one(out, *first);
  }
}

template <typename Container>
void serialize_container(std::ostream& out, const Container& container) {
  serialize_range(out, std::begin(container), std::end(container));
}

// Deserialize a range [first, last) of objects using iterators
template <typename Iterator>
void deserialize_range(std::istream& in, Iterator first, Iterator last) {
  for (; first != last; ++first) {
    deserialize_one(in, *first);
  }
}

// Helper for containers with .begin() and .end()
template <typename Container>
void deserialize_container(std::istream& in, Container& container) {
  deserialize_range(in, std::begin(container), std::end(container));
}

// // class to indicate a non-serialisable type
// struct NonSerializable {};

}  // namespace kakuhen::util::serialize
