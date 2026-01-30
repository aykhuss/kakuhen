#pragma once

#include <cstdint>

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
