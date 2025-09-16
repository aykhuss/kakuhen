#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace kakuhen::util {
/// simple FNV-1a hashing

using HashValue_t = std::uint64_t;

class Hash {
 public:
  Hash() : hash_(FNV_OFFSET_BASIS) {}

  template <typename T>
  Hash& add(const T& value) noexcept {
    add_one(hash_, value);
    return *this;
  }

  template <typename T>
  Hash& add(const T* data, size_t count) noexcept {
    add_array(hash_, data, count);
    return *this;
  }

  template <typename T>
  Hash& add(const std::vector<T>& vec) noexcept {
    add_vector(hash_, vec);
    return *this;
  }

  Hash& add(const std::string& str) noexcept {
    add_string(hash_, str);
    return *this;
  }

  inline HashValue_t value() const noexcept { return hash_; }

  inline void reset() noexcept { hash_ = FNV_OFFSET_BASIS; }

  std::string encode_hex() const noexcept {
    std::ostringstream oss;
    oss << std::hex << hash_;
    return oss.str();
  }

  //> Helper functions that can also be used outside the class

  static constexpr void add_bytes(HashValue_t& hash, const void* data,
                                  size_t len) noexcept {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
      hash ^= bytes[i];
      hash *= FNV_PRIME;
    }
  }

  //> Hash a single POD value
  template <typename T>
  static inline void add_one(HashValue_t& hash, const T& value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>,
                  "add_one: Hashing only supports trivially copyable types");
    add_bytes(hash, &value, sizeof(T));
  }

  //> Hash a contiguous array of trivially copyable values
  template <typename T>
  static inline void add_array(HashValue_t& hash, const T* data,
                               size_t count) noexcept {
    static_assert(std::is_trivially_copyable_v<T>,
                  "add_array: Hashing only supports trivially copyable types");
    add_bytes(hash, data, count * sizeof(T));
  }

  //> Hash a vector of PODs
  //@todo: generalize to any container of PODs (-> iterators)
  template <typename T>
  static inline void add_vector(HashValue_t& hash,
                                const std::vector<T>& vec) noexcept {
    static_assert(std::is_trivially_copyable_v<T>,
                  "add_vector: Hashing only supports trivially copyable types");
    add_bytes(hash, vec.data(), vec.size() * sizeof(T));
  }

  //> Hash a string
  static inline void add_string(HashValue_t& hash,
                                const std::string& str) noexcept {
    add_bytes(hash, str.data(), str.size());
  }

 private:
  static constexpr HashValue_t FNV_OFFSET_BASIS = 14695981039346656037ull;
  static constexpr HashValue_t FNV_PRIME = 1099511628211ull;
  HashValue_t hash_;
};

}  // namespace kakuhen::util
