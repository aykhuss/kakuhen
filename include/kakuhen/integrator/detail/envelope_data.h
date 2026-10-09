#pragma once

/// @file envelope_data.h
/// @brief pending envelope observations and the shared envelope update rule
///
/// The envelope R(x) is a product of `ndim` positive factors, one from each
/// group of the flat factor table. `collect_envelope` records every |f| that
/// exceeds the (fixed) envelope in an `EnvelopeData` batch; `adapt_envelope`
/// and `merge_envelope` later replay the batch with `raise_envelope_record`,
/// the same update `raise_envelope` applies right away.

#include "kakuhen/integrator/integral_accumulator.h"
#include "kakuhen/ndarray/ndarray.h"
#include "kakuhen/util/serialize.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace kakuhen::integrator::detail {

/*!
 * @brief batch of envelope violations plus the statistics of all its evaluations
 *
 * Record `i` is the value `values[i]` with its `ndim` flat factor indices
 * `indices[i * ndim, (i + 1) * ndim)`. Stored as two contiguous arrays
 * (structure of arrays), so appending doesn't allocate per record and a
 * replay walks both arrays sequentially. Records keep their collection
 * order, which the replay depends on.
 *
 * `acc` covers every evaluation of the batch, not only the recorded ones, and
 * feeds the absolute-integral estimate when the batch is merged. A non-finite
 * sample enters it as |f| = 0.
 *
 * Invariants (checked by `deserialize`):
 *  - `indices.size() == values.size() * ndim`
 *  - `values.size() <= acc.count()`
 *  - every value is positive with a finite square
 *
 * @tparam T value type (e.g. double)
 * @tparam U count type (e.g. uint64_t)
 * @tparam S size/index type of the factor table
 */
template <typename T, typename U, typename S>
struct EnvelopeData {
  static_assert(std::is_floating_point_v<T>);
  static_assert(std::is_trivially_copyable_v<S>, "append relies on a nothrow range insert");

  IntegralAccumulator<T, U> acc{};  //!< statistics of |f| over all evaluations in the batch
  std::vector<T> values;            //!< |f| of each violating observation, in order
  std::vector<S> indices;           //!< flat factor indices, `ndim` per observation

  /// @brief whether the batch holds any evaluation (even without violations)
  [[nodiscard]] bool present() const noexcept {
    return acc.count() != U(0);
  }

  /// @brief drop all evaluations and records; keeps the allocated capacity
  void clear() noexcept {
    acc.reset();
    values.clear();
    indices.clear();
  }

  /// @brief the `ndim` flat factor indices of record `i`
  [[nodiscard]] std::span<const S> cells(std::size_t i, S ndim) const noexcept {
    assert((i + 1) * ndim <= indices.size());
    return {indices.data() + i * ndim, static_cast<std::size_t>(ndim)};
  }

  /*!
   * @brief append one violating observation
   *
   * Strong exception guarantee: if an allocation fails, both arrays are
   * unchanged. Doesn't touch `acc`; the caller accounts for the evaluation
   * separately.
   *
   * @param value the violating |f|
   * @param cells the `ndim` flat factor indices of the observation
   */
  void append(T value, std::span<const S> cells) {
    // a range insert of trivially copyable elements at the end is strong
    indices.insert(indices.end(), cells.begin(), cells.end());
    try {
      values.push_back(value);
    } catch (...) {
      indices.resize(indices.size() - cells.size());  // shrinking does not allocate
      throw;
    }
  }

  /*!
   * @brief write the batch to a stream
   *
   * Layout: a presence flag (`uint8_t`, 0 or 1), followed if present by `acc`,
   * the record count (`U`) and the records, each as its value followed by its
   * `ndim` indices.
   *
   * @param out output stream
   * @param ndim number of indices per record
   */
  void serialize(std::ostream& out, S ndim) const {
    using namespace util::serialize;
    assert(indices.size() == values.size() * static_cast<std::size_t>(ndim));
    serialize_one<uint8_t>(out, present() ? uint8_t(1) : uint8_t(0));
    if (!present()) return;
    acc.serialize(out);
    serialize_one<U>(out, static_cast<U>(values.size()));
    for (std::size_t i = 0; i < values.size(); ++i) {
      serialize_one<T>(out, values[i]);
      for (const S index : cells(i, ndim))
        serialize_one<S>(out, index);
    }
  }

  /*!
   * @brief read and validate a batch written by `serialize`
   *
   * Records are appended one at a time, so an untrusted record count never
   * causes a large upfront allocation; a truncated stream fails on the first
   * missing record instead.
   *
   * @param in input stream
   * @param ndim number of indices per record, must be > 0
   * @param table_size size of the factor table, made up of `ndim` equally
   *        sized factor groups (`table_size = ndim * group_size`); must be a
   *        positive multiple of `ndim`
   *          VEGAS:  `group_size = ndiv`
   *          BASIN:  `group_size = ndiv1 * ndiv2 == ndiv0`
   *        so we can check that cells[j] is in [j*group_size, (j+1)*group_size)
   * @return the batch, empty if the stream holds none
   * @throws std::runtime_error if the flag, statistics, record count, a value
   *         or an index is invalid (the index of a record's `j`-th dimension
   *         has to lie in the `j`-th factor group)
   */
  [[nodiscard]] static EnvelopeData deserialize(std::istream& in, S ndim, S table_size) {
    using namespace util::serialize;
    assert(ndim > S(0) && table_size >= ndim && table_size % ndim == S(0));
    EnvelopeData data;
    uint8_t flag = 0;
    deserialize_one<uint8_t>(in, flag);
    if (flag > uint8_t(1)) throw std::runtime_error("invalid envelope batch flag");
    if (flag == uint8_t(0)) return data;

    data.acc.deserialize(in);
    U count = 0;
    deserialize_one<U>(in, count);
    // a present batch has at least one evaluation
    if (!data.acc.is_finite() || data.acc.count() == U(0) || data.acc.value() < T(0) ||
        data.acc.f2_.result() < T(0) || count > data.acc.count()) {
      throw std::runtime_error("invalid envelope batch statistics or size");
    }

    const S group_size = table_size / ndim;
    std::vector<S> cells(ndim);
    for (U i = 0; i < count; ++i) {
      T value{};
      deserialize_one<T>(in, value);
      // negated so that NaN is rejected; collection rejects values with an overflowing square
      if (!(value > T(0)) || !std::isfinite(value * value)) {
        throw std::runtime_error("invalid envelope observation value");
      }
      for (S j = 0; j < ndim; ++j) {
        deserialize_one<S>(in, cells[j]);
        if (cells[j] >= table_size || cells[j] / group_size != j) {
          throw std::runtime_error("invalid envelope observation factor index");
        }
      }
      data.append(value, cells);
    }
    return data;
  }
};

/*!
 * @brief envelope R at a point, i.e. the product of the factors at `indices`
 *
 * @param table flat factor table
 * @param indices flat factor indices of the point, one per dimension
 */
template <typename T, typename S>
[[nodiscard]] T envelope_bound(const ndarray::NDArray<T, S>& table, std::span<const S> indices) {
  T bound = T(1);
  for (const S index : indices)
    bound *= table[index];
  return bound;
}

/*!
 * @brief raise the factors at `indices` by `gamma` if `value` exceeds their product
 *
 * The one bounded update shared by immediate training (`raise_envelope`) and
 * deferred replay (`adapt_envelope`, `merge_envelope`): at most one raise per
 * observation, even if a single raise doesn't cover `value`.
 *
 * @param table flat factor table to update
 * @param indices flat factor indices of the observation
 * @param value the observed |f|
 * @param gamma raising factor, must be > 1
 * @return whether the factors were raised, i.e. whether `value` was a violation
 */
template <typename T, typename S>
[[nodiscard]] bool raise_envelope_record(ndarray::NDArray<T, S>& table, std::span<const S> indices,
                                         T value, T gamma) {
  if (value <= envelope_bound(table, indices)) return false;
  for (const S index : indices)
    table[index] *= gamma;
  return true;
}

}  // namespace kakuhen::integrator::detail
