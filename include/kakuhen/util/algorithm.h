#pragma once

#include <algorithm>
#include <iterator>

namespace kakuhen::util::algorithm {

/*!
 * @brief Algorithms and utilities.
 */

/**
 * @brief Branch-friendly lower_bound variant using a "staircase" search.
 *
 * This is an alternative to `std::lower_bound` that can reduce branch
 * mispredictions on some workloads by keeping the search direction monotonic.
 * It requires random-access iterators for efficient indexing.
 *
 * @note Based on: https://mhdm.dev/posts/sb_lower_bound/
 *
 * @tparam RandomIt Random-access iterator type.
 * @tparam T Value type to compare.
 * @tparam Compare Strict weak ordering comparator.
 * @param first Iterator to the first element.
 * @param last Iterator to one past the last element.
 * @param value The value to search for.
 * @param comp Comparator such that `comp(a, b)` is true if `a < b`.
 * @return Iterator to the first element not less than `value`.
 */
template <class RandomIt, class T, class Compare>
[[nodiscard]] constexpr RandomIt lower_bound(RandomIt first, RandomIt last, const T& value,
                                              Compare comp) {
  auto length = last - first;
  while (length > 0) {
    auto half = length / 2;
    if (comp(first[half], value)) {
      first += length - half;
    }
    length = half;
  }
  return first;
}

/**
 * @brief Branch-friendly upper_bound variant using a "staircase" search.
 *
 * This is an alternative to `std::upper_bound` that can reduce branch
 * mispredictions on some workloads by keeping the search direction monotonic.
 * It requires random-access iterators for efficient indexing.
 *
 * @tparam RandomIt Random-access iterator type.
 * @tparam T Value type to compare.
 * @tparam Compare Strict weak ordering comparator.
 * @param first Iterator to the first element.
 * @param last Iterator to one past the last element.
 * @param value The value to search for.
 * @param comp Comparator such that `comp(a, b)` is true if `a < b`.
 * @return Iterator to the first element greater than `value`.
 */
template <class RandomIt, class T, class Compare>
[[nodiscard]] constexpr RandomIt upper_bound(RandomIt first, RandomIt last, const T& value,
                                              Compare comp) {
  auto length = last - first;
  while (length > 0) {
    auto half = length / 2;
    if (!comp(value, first[half])) {
      first += length - half;
    }
    length = half;
  }
  return first;
}

/**
 * @brief lower_bound with a starting hint using exponential search.
 *
 * Performs an exponential search around the hint to quickly bound the range,
 * then performs a branch-friendly binary search. This is significantly faster
 * than a full search when the hint is near the correct position.
 *
 * @tparam RandomIt Random-access iterator type.
 * @tparam T Value type to compare.
 * @tparam Compare Strict weak ordering comparator.
 * @param first Iterator to the first element.
 * @param last Iterator to one past the last element.
 * @param hint Hint iterator (clamped to [first, last-1] if necessary).
 * @param value The value to search for.
 * @param comp Comparator such that `comp(a, b)` is true if `a < b`.
 * @return Iterator to the first element not less than `value`.
 */
template <class RandomIt, class T, class Compare>
[[nodiscard]] constexpr RandomIt lower_bound_with_hint(RandomIt first, RandomIt last, RandomIt hint,
                                                       const T& value, Compare comp) {
  if (first == last) return first;

  using diff_t = typename std::iterator_traits<RandomIt>::difference_type;
  const diff_t len = last - first;

  // Clamp hint to valid index range [0, len-1]
  diff_t hint_idx = hint - first;
  if (hint_idx < 0) hint_idx = 0;
  if (hint_idx >= len) hint_idx = len - 1;
  hint = first + hint_idx;

  if (comp(*hint, value)) {
    // Value is in (hint, last)
    diff_t step = 1;
    diff_t lo = hint_idx + 1;
    diff_t hi = hint_idx + 1;
    if (lo >= len) return last;

    while (hi < len && comp(*(first + hi), value)) {
      lo = hi + 1;
      step *= 2;
      hi = hint_idx + step;
      if (hi >= len) {
        hi = len - 1;
        if (comp(*(first + hi), value)) return last;
        break;
      }
    }
    return ::kakuhen::util::algorithm::lower_bound(first + lo, first + hi + 1, value, comp);
  } else {
    // Value is in [first, hint]
    if (hint == first || comp(*(hint - 1), value)) return hint;

    diff_t step = 1;
    diff_t hi = hint_idx;
    diff_t lo = hint_idx;

    while (lo > 0) {
      diff_t offset = std::min(step, lo);
      lo -= offset;
      if (comp(*(first + lo), value)) {
        lo += 1;
        break;
      }
      hi = lo;
      step *= 2;
    }
    return ::kakuhen::util::algorithm::lower_bound(first + lo, first + hi + 1, value, comp);
  }
}

/**
 * @brief upper_bound with a starting hint using exponential search.
 *
 * @tparam RandomIt Random-access iterator type.
 * @tparam T Value type to compare.
 * @tparam Compare Strict weak ordering comparator.
 * @param first Iterator to the first element.
 * @param last Iterator to one past the last element.
 * @param hint Hint iterator (clamped to [first, last-1] if necessary).
 * @param value The value to search for.
 * @param comp Comparator such that `comp(a, b)` is true if `a < b`.
 * @return Iterator to the first element greater than `value`.
 */
template <class RandomIt, class T, class Compare>
[[nodiscard]] constexpr RandomIt upper_bound_with_hint(RandomIt first, RandomIt last, RandomIt hint,
                                                       const T& value, Compare comp) {
  if (first == last) return first;

  using diff_t = typename std::iterator_traits<RandomIt>::difference_type;
  const diff_t len = last - first;

  diff_t hint_idx = hint - first;
  if (hint_idx < 0) hint_idx = 0;
  if (hint_idx >= len) hint_idx = len - 1;
  hint = first + hint_idx;

  if (!comp(value, *hint)) {
    // Value is in (hint, last)
    diff_t step = 1;
    diff_t lo = hint_idx + 1;
    diff_t hi = hint_idx + 1;
    if (lo >= len) return last;

    while (hi < len && !comp(value, *(first + hi))) {
      lo = hi + 1;
      step *= 2;
      hi = hint_idx + step;
      if (hi >= len) {
        hi = len - 1;
        if (!comp(value, *(first + hi))) return last;
        break;
      }
    }
    return ::kakuhen::util::algorithm::upper_bound(first + lo, first + hi + 1, value, comp);
  } else {
    // Value is in [first, hint]
    if (hint == first || !comp(value, *(hint - 1))) return hint;

    diff_t step = 1;
    diff_t hi = hint_idx;
    diff_t lo = hint_idx;

    while (lo > 0) {
      diff_t offset = std::min(step, lo);
      lo -= offset;
      if (!comp(value, *(first + lo))) {
        lo += 1;
        break;
      }
      hi = lo;
      step *= 2;
    }
    return ::kakuhen::util::algorithm::upper_bound(first + lo, first + hi + 1, value, comp);
  }
}

}  // namespace kakuhen::util::algorithm
