#pragma once

#include <cstdint>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace kakuhen::util {

/// @brief 64-bit unsigned integer type used for storing FNV-1a hash values.
using HashValue_t = std::uint64_t;

/*!
 * @brief Implements the FNV-1a non-cryptographic hash function (64-bit).
 *
 * This class provides a convenient interface for computing FNV-1a hashes
 * for various data types. It processes data byte-by-byte.
 *
 * @note This implementation hashes the raw memory representation of types.
 * Therefore, hashes of multi-byte integers are endian-dependent and should
 * not be expected to match across big-endian and little-endian systems.
 *
 * @see https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
 */
class Hash {
 public:
  /// @brief Initialize hash with the FNV offset basis.
  constexpr Hash() noexcept : hash_(FNV_OFFSET_BASIS) {}

  /*!
   * @brief Adds a single trivially copyable value to the hash.
   *
   * @tparam T The type of the value to add (must be trivially copyable).
   * @param value The value to add to the hash.
   * @return A reference to the current `Hash` object for chaining calls.
   */
  template <typename T>
  Hash& add(const T& value) noexcept {
    add_one(hash_, value);
    return *this;
  }

  /*!
   * @brief Adds a C-style array of trivially copyable values to the hash.
   *
   * @tparam T The type of the array elements (must be trivially copyable).
   * @param data A pointer to the beginning of the array.
   * @param count The number of elements in the array.
   * @return A reference to the current `Hash` object for chaining calls.
   */
  template <typename T>
  Hash& add(const T* data, size_t count) noexcept {
    add_span(hash_, std::span<const T>(data, count));
    return *this;
  }

  /*!
   * @brief Adds a contiguous sequence of values (span) to the hash.
   *
   * This overload handles `std::vector`, `std::array`, and C-arrays implicitly.
   *
   * @tparam T The type of the elements (must be trivially copyable).
   * @param data The span of data to add.
   * @return A reference to the current `Hash` object for chaining calls.
   */
  template <typename T>
  Hash& add(std::span<const T> data) noexcept {
    add_span(hash_, data);
    return *this;
  }

  /*!
   * @brief Adds a `std::vector` of trivially copyable values to the hash.
   *
   * This overload enables implicit usage with temporary vectors, where span deduction might fail.
   *
   * @tparam T The type of the vector elements (must be trivially copyable).
   * @param vec The `std::vector` to add to the hash.
   * @return A reference to the current `Hash` object for chaining calls.
   */
  template <typename T>
  Hash& add(const std::vector<T>& vec) noexcept {
    return add(std::span<const T>(vec));
  }

  /*!
   * @brief Adds a string view to the hash.
   *
   * @param str The string to add.
   * @return A reference to the current `Hash` object for chaining calls.
   */
  Hash& add(std::string_view str) noexcept {
    add_string(hash_, str);
    return *this;
  }

  /*!
   * @brief Gets the current hash value.
   *
   * @return The current 64-bit FNV-1a hash value.
   */
  [[nodiscard]] inline constexpr HashValue_t value() const noexcept {
    return hash_;
  }

  /*!
   * @brief Resets the hash to its initial FNV offset basis.
   */
  inline void reset() noexcept {
    hash_ = FNV_OFFSET_BASIS;
  }

  /*!
   * @brief Encodes the current hash value as a lowercase hexadecimal string.
   *
   * @return A 16-character string representation of the hash.
   */
  [[nodiscard]] std::string encode_hex() const noexcept {
    // // this is faster but I don't care much about performance for the hex encoding
    // constexpr char hex_digits[] = "0123456789abcdef";
    // std::string s(16, '0');
    // for (size_t i = 0; i < 16; ++i) {
    //   // Extract 4 bits at a time, starting from the most significant
    //   s[i] = hex_digits[(hash_ >> (60 - 4 * i)) & 0xF];
    // }
    // return s;
    std::ostringstream oss;
    oss << std::hex << hash_;
    return oss.str();
  }

  // Helper functions that can also be used outside the class

  /*!
   * @brief Adds a raw byte array to the FNV-1a hash.
   *
   * @param hash The current hash value to update.
   * @param data A pointer to the byte array.
   * @param len The number of bytes to hash.
   */
  static constexpr void add_bytes(HashValue_t& hash, const void* data, size_t len) noexcept {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
      hash ^= bytes[i];
      hash *= FNV_PRIME;
    }
  }

  /*!
   * @brief Hashes a single trivially copyable value.
   *
   * @tparam T The type of the value (must be trivially copyable).
   * @param hash The current hash value to update.
   * @param value The value to hash.
   */
  template <typename T>
  static inline void add_one(HashValue_t& hash, const T& value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>,
                  "add_one: Hashing only supports trivially copyable types");
    add_bytes(hash, &value, sizeof(T));
  }

  /*!
   * @brief Hashes a span of trivially copyable values.
   *
   * @tparam T The type of the elements (must be trivially copyable).
   * @param hash The current hash value to update.
   * @param data The span of data to hash.
   */
  template <typename T>
  static inline void add_span(HashValue_t& hash, std::span<const T> data) noexcept {
    static_assert(std::is_trivially_copyable_v<T>,
                  "add_span: Hashing only supports trivially copyable types");
    add_bytes(hash, data.data(), data.size_bytes());
  }

  /*!
   * @brief Hashes a string view.
   *
   * @param hash The current hash value to update.
   * @param str The string view to hash.
   */
  static inline void add_string(HashValue_t& hash, std::string_view str) noexcept {
    add_bytes(hash, str.data(), str.size());
  }

 private:
  static constexpr HashValue_t FNV_OFFSET_BASIS = 14695981039346656037ull;
  static constexpr HashValue_t FNV_PRIME = 1099511628211ull;
  HashValue_t hash_;
};

}  // namespace kakuhen::util
