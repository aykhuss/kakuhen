#pragma once

#include "kakuhen/util/hash.h"
#include "kakuhen/util/serialize.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <format>
#include <istream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>

namespace kakuhen::integrator::detail {

/*!
 * @brief Checks that `row` holds the upper bin boundaries of a 1D grid on [0, 1].
 *
 * The boundaries must be non-decreasing, start at >= 0, and end at exactly 1.
 * Repeated boundaries are accepted as they can arise from rounding during adaptation.
 *
 * @param row The bin boundaries.
 * @param who The integrator name prefixed to the error message.
 * @throws std::runtime_error if the boundaries are invalid (incl. NaN).
 */
template <typename T>
void validate_grid_row(std::span<const T> row, std::string_view who) {
  const auto out_of_order = [](T lo, T hi) { return !(lo <= hi); };  // true also for NaN
  if (row.empty() || !(row.front() >= T(0)) || row.back() != T(1) ||
      std::ranges::adjacent_find(row, out_of_order) != row.end()) {
    throw std::runtime_error(std::format("{}: corrupt state (invalid grid boundaries)", who));
  }
}

/*!
 * @brief Merges a serialized data block into the accumulated data of an integrator.
 *
 * Reads the grid hash, the integral accumulator, the adaptation count, and the
 * grid cells. Everything is validated before any state is modified (strong
 * exception guarantee). The cells form contiguous rows of `row_size` that each
 * partition the adaptation samples, so the counts of every row must add up to
 * the adaptation count. Together with the checked total count, this also rules
 * out an overflow of the merged cell counts.
 *
 * @pre The destination already satisfies the row-count invariant; `row_size`
 *      is positive and divides `cells.size()`; assigning the result and
 *      accumulating cells cannot throw.
 *
 * @param in The input stream, positioned after the grid dimensions.
 * @param who The integrator name prefixed to error messages.
 * @param hash The hash of the current grid that the data must have been sampled on.
 * @param result The integral accumulator to merge into.
 * @param count The adaptation count to merge into.
 * @param cells The grid accumulators to merge into.
 * @param row_size The number of cells per row.
 * @throws std::runtime_error if the stream is truncated, corrupt, or sampled on a different grid.
 * @throws std::overflow_error if the merged sums or counts overflow.
 */
template <typename IntAcc, std::unsigned_integral U, typename Cells>
void merge_data_stream(std::istream& in, std::string_view who, util::HashValue_t hash,
                       IntAcc& result, U& count, Cells& cells, typename Cells::size_type row_size) {
  static_assert(std::same_as<typename IntAcc::count_type, U> &&
                    std::same_as<typename Cells::value_type::count_type, U>,
                "all counts must share the type the overflow checks are performed in");
  using util::serialize::deserialize_one;
  const auto fail = [who](std::string_view what) {
    throw std::runtime_error(std::format("{}: {}", who, what));
  };
  const auto overflow = [who](std::string_view what) {
    throw std::overflow_error(std::format("{}: {} overflows", who, what));
  };

  util::HashValue_t hash_in;
  deserialize_one(in, hash_in);
  if (hash_in != hash) fail("incompatible data (grid hash mismatch)");
  IntAcc result_in;
  result_in.deserialize(in);
  const auto sum = result_in.f_.result();
  const auto sum_squares = result_in.f2_.result();
  if (!result_in.is_finite() || sum_squares < 0 ||
      (result_in.count() == 0 && (sum != 0 || sum_squares != 0))) {
    fail("corrupt data (invalid integral accumulator)");
  }
  U count_in;
  deserialize_one(in, count_in);

  constexpr U MAX_COUNT = std::numeric_limits<U>::max();
  if (result_in.count() > MAX_COUNT - result.count()) overflow("merged sample count");
  if (count_in > MAX_COUNT - count) overflow("merged adaptation count");
  Cells cells_in;
  cells_in.deserialize_expected_shape(in, cells.shape());

  assert(row_size > 0 && cells.size() % row_size == 0);
  for (typename Cells::size_type row = 0; row < cells.size(); row += row_size) {
    U remaining_count = count_in;
    for (auto i = row; i < row + row_size; ++i) {
      const auto& cell = cells_in[i];
      if (!std::isfinite(cell.value()) || cell.value() < 0 || cell.count() > remaining_count ||
          (cell.count() == 0 && cell.value() != 0)) {
        fail("corrupt data (invalid grid accumulator)");
      }
      remaining_count -= cell.count();
      if (!std::isfinite(cells[i].value() + cell.value())) overflow("merged grid accumulator");
    }
    if (remaining_count != 0) fail("corrupt data (grid counts do not match the adaptation count)");
  }
  IntAcc merged = result;
  merged.accumulate(result_in);
  if (!merged.is_finite()) overflow("merged integral");

  // commit (cannot throw)
  result = merged;
  count += count_in;
  for (typename Cells::size_type i = 0; i < cells.size(); ++i) {
    cells[i].accumulate(cells_in[i]);
  }
}

}  // namespace kakuhen::integrator::detail
