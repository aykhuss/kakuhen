#pragma once

namespace kakuhen::util::algorithm {

/*!
 * @brief Algorithms and utilities.
 */

/*!
 * @brief Branch-friendly lower_bound variant using a "staircase" search.
 *
 * This is an alternative to `std::lower_bound` that can reduce branch
 * mispredictions on some workloads by keeping the search direction monotonic.
 * It requires random-access iterators because it uses indexing and subtraction.
 *
 * @note Based on: https://mhdm.dev/posts/sb_lower_bound/
 *
 * @tparam ForwardIt Random-access iterator type.
 * @tparam T Value type to compare.
 * @tparam Compare Strict weak ordering comparator.
 * @param first Iterator to the first element.
 * @param last Iterator to one past the last element.
 * @param value The value to search for.
 * @param comp Comparator such that `comp(a, b)` is true if `a < b`.
 * @return Iterator to the first element not less than `value`.
 */
template <class ForwardIt, class T, class Compare>
constexpr ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp) {
  auto length = last - first;
  while (length > 0) {
    auto half = length / 2;
    if (comp(first[half], value)) {
      // length - half equals half + length % 2
      first += length - half;
    }
    length = half;
  }
  return first;
}

/*!
 * @brief Branch-friendly upper_bound variant using a "staircase" search.
 *
 * This is an alternative to `std::upper_bound` that can reduce branch
 * mispredictions on some workloads by keeping the search direction monotonic.
 * It requires random-access iterators because it uses indexing and subtraction.
 *
 * @note Based on: https://mhdm.dev/posts/sb_lower_bound/
 *
 * @tparam ForwardIt Random-access iterator type.
 * @tparam T Value type to compare.
 * @tparam Compare Strict weak ordering comparator.
 * @param first Iterator to the first element.
 * @param last Iterator to one past the last element.
 * @param value The value to search for.
 * @param comp Comparator such that `comp(a, b)` is true if `a < b`.
 * @return Iterator to the first element greater than `value`.
 */
template <class ForwardIt, class T, class Compare>
constexpr ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp) {
  auto length = last - first;
  while (length > 0) {
    auto half = length / 2;
    if (!comp(value, first[half])) {
      // length - half equals half + length % 2
      first += length - half;
    }
    length = half;
  }
  return first;
}

}  // namespace kakuhen::util::algorithm
