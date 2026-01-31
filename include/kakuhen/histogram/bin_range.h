#pragma once

#include <cstdint>
#include <string_view>

namespace kakuhen::histogram {

/**
 * @brief Identifiers for different types of histogram bins.
 */
enum class BinKind : uint8_t {
  Regular = 0,    //!< A standard bin within the axis range.
  Underflow = 1,  //!< Bin for values below the axis range.
  Overflow = 2,   //!< Bin for values above the axis range.
  Invalid = 3     //!< Sentinel for invalid or uninitialized bins.
};

/**
 * @brief Converts a BinKind to its string representation.
 * @param kind The BinKind to convert.
 * @return A string view of the bin kind name.
 */
[[nodiscard]] constexpr std::string_view to_string(BinKind kind) noexcept {
  switch (kind) {
    case BinKind::Regular:
      return "Regular";
    case BinKind::Underflow:
      return "Underflow";
    case BinKind::Overflow:
      return "Overflow";
    case BinKind::Invalid:
      return "Invalid";
  }
  return "Unknown";
}

/**
 * @brief Represents the physical boundaries and classification of a single bin.
 * @tparam T The coordinate value type.
 */
template <typename T>
struct BinRange {
  BinKind kind;  //!< The classification of the bin.
  T low;         //!< Lower boundary of the bin.
  T upp;         //!< Upper boundary of the bin.
};

}  // namespace kakuhen::histogram
