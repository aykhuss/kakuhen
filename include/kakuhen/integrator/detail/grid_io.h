#pragma once

#include "kakuhen/util/hash.h"
#include "kakuhen/util/serialize.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <istream>
#include <limits>
#include <span>
#include <stdexcept>

namespace kakuhen::integrator::detail {

/*!
 * @brief check that `row` holds the upper bin boundaries of a 1D grid on [0, 1]
 *
 * The boundaries must be non-decreasing, start at >= 0 and end at exactly 1.
 * Repeated boundaries are fine, as rounding during adaptation can produce them.
 *
 * @param row bin boundaries
 * @throws std::runtime_error if the boundaries are invalid (incl. NaN)
 */
template <typename T>
void validate_grid_row(std::span<const T> row) {
  const auto out_of_order = [](T lo, T hi) { return !(lo <= hi); };  // true also for NaN
  if (row.empty() || !(row.front() >= T(0)) || row.back() != T(1) ||
      std::ranges::adjacent_find(row, out_of_order) != row.end()) {
    throw std::runtime_error("corrupt state (invalid grid boundaries)");
  }
}

/*!
 * @brief merge a serialized data block into the accumulated data of an integrator
 *
 * Reads the grid hash, integral accumulator, adaptation count and grid cells.
 * Everything is validated before any state changes (strong exception
 * guarantee). The cells form contiguous rows of `row_size` that each
 * partition the adaptation samples, so the counts of every row must add up to
 * the adaptation count. Together with the checked total count, this also
 * rules out an overflow of the merged cell counts.
 *
 * @pre the destination already satisfies the row-count invariant; `row_size`
 *      is positive and divides `cells.size()`; assigning the result and
 *      accumulating cells can't throw
 *
 * @param in input stream, positioned after the grid dimensions
 * @param hash hash of the current grid the data must have been sampled on
 * @param result integral accumulator to merge into
 * @param count adaptation count to merge into
 * @param cells grid accumulators to merge into
 * @param row_size number of cells per row
 * @throws std::runtime_error if the stream is truncated, corrupt or sampled on a different grid
 * @throws std::overflow_error if the merged sums or counts overflow
 */
template <typename IntAcc, std::unsigned_integral U, typename Cells>
void merge_data_stream(std::istream& in, util::HashValue_t hash, IntAcc& result, U& count,
                       Cells& cells, typename Cells::size_type row_size) {
  static_assert(std::same_as<typename IntAcc::count_type, U> &&
                    std::same_as<typename Cells::value_type::count_type, U>,
                "all counts must share the type the overflow checks are performed in");
  using util::serialize::deserialize_one;
  util::HashValue_t hash_in;
  deserialize_one(in, hash_in);
  if (hash_in != hash) throw std::runtime_error("incompatible data (grid hash mismatch)");
  IntAcc result_in;
  result_in.deserialize(in);
  const auto sum = result_in.f_.result();
  const auto sum_squares = result_in.f2_.result();
  if (!result_in.is_finite() || sum_squares < 0 ||
      (result_in.count() == 0 && (sum != 0 || sum_squares != 0))) {
    throw std::runtime_error("corrupt data (invalid integral accumulator)");
  }
  U count_in;
  deserialize_one(in, count_in);

  constexpr U MAX_COUNT = std::numeric_limits<U>::max();
  if (result_in.count() > MAX_COUNT - result.count()) {
    throw std::overflow_error("merged sample count overflows");
  }
  if (count_in > MAX_COUNT - count) {
    throw std::overflow_error("merged adaptation count overflows");
  }
  Cells cells_in;
  cells_in.deserialize_expected_shape(in, cells.shape());

  assert(row_size > 0 && cells.size() % row_size == 0);
  for (typename Cells::size_type row = 0; row < cells.size(); row += row_size) {
    U remaining_count = count_in;
    for (auto i = row; i < row + row_size; ++i) {
      const auto& cell = cells_in[i];
      if (!std::isfinite(cell.value()) || cell.value() < 0 || cell.count() > remaining_count ||
          (cell.count() == 0 && cell.value() != 0)) {
        throw std::runtime_error("corrupt data (invalid grid accumulator)");
      }
      remaining_count -= cell.count();
      if (!std::isfinite(cells[i].value() + cell.value())) {
        throw std::overflow_error("merged grid accumulator overflows");
      }
    }
    if (remaining_count != 0) {
      throw std::runtime_error("corrupt data (grid counts do not match the adaptation count)");
    }
  }
  IntAcc merged = result;
  merged.accumulate(result_in);
  if (!merged.is_finite()) throw std::overflow_error("merged integral overflows");

  // commit (cannot throw)
  result = merged;
  count += count_in;
  for (typename Cells::size_type i = 0; i < cells.size(); ++i) {
    cells[i].accumulate(cells_in[i]);
  }
}

}  // namespace kakuhen::integrator::detail
