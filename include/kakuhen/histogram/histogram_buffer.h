#pragma once

#include "kakuhen/histogram/histogram_data.h"
#include "kakuhen/util/accumulator.h"
#include "kakuhen/util/numeric_traits.h"
#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kakuhen::histogram {

/*!
 * @brief A high-performance thread-local buffer for histogram filling.
 *
 * This class implements a sparse-dense buffering strategy to efficiently
 * accumulate weights into a global histogram storage. It is designed to
 * minimize memory contention and numerical errors during high-frequency
 * filling in Monte Carlo simulations.
 *
 * Key Features:
 * - **TwoSum Compensation (Default)**: Uses compensated summation internally via
 *   the `Acc` type to handle large cancellations (e.g., +W and -W) before they
 *   touch global memory. This is configurable for performance.
 * - **Packed Generation Index**: Uses a "Sparse Set" approach with a packed
 *   generation index to perform O(1) validity checks without zeroing memory
 *   between events.
 * - **Event-Level Consolidation**: Accumulates all weights for a specific bin
 *   within a single event linearly. This ensures that large cancellations
 *   (e.g., +10.0 and -9.9) result in a small net weight (0.1) which is then
 *   squared for error estimation (0.01), rather than summing squares independently.
 *
 * @tparam NT The numeric traits defining value type and index type.
 * @tparam Acc The accumulator type for intra-event summation. Defaults to `TwoSum`.
 *             Use `NaiveAccumulator` for maximum speed if cancellations are not expected.
 */
template <typename NT = util::num_traits_t<>,
          typename Acc = kakuhen::util::accumulator::Accumulator<typename NT::value_type>>
class HistogramBuffer {
 public:
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  using count_type = typename num_traits::count_type;
  using acc_type = Acc;
  // shorthands
  using S = size_type;
  using T = value_type;
  using U = count_type;

  /**
   * @brief Initializes and configures the buffer for a specific global storage size.
   *
   * This method must be called once before any filling occurs. It calculates the optimal
   * bit allocation for the packed generation index based on the total number of global bins
   * to maximize the time between mandatory map clears.
   *
   * @param n_total_bins The total number of bins in the global `HistogramData` storage.
   * @param reserve_size Initial capacity for the thread-local dense buffers to avoid reallocations.
   * @throws std::runtime_error If the index type `S` is too small to accommodate the bin
   *         index and the minimum required generation counter bits (4).
   *
   * @note The `sparse_map_` is initialized with zeros. A value of 0 in the map indicates
   *       that the bin has never been touched in the current generation.
   */
  void init(S n_total_bins, std::size_t reserve_size = 1024) {
    if (n_total_bins == 0) return;

    // Determine total bits available in the index type S
    constexpr unsigned int total_bits = std::numeric_limits<S>::digits;

    // Calculate bits needed for the buffer index
    const unsigned int index_bits =
        static_cast<unsigned int>(std::bit_width(static_cast<std::make_unsigned_t<S>>(n_total_bins)));

    // Ensure we have at least 4 bits for generation index
    constexpr unsigned int min_gen_bits = 4;
    if (total_bits < index_bits + min_gen_bits) {
      throw std::runtime_error("Index type S is too small to support " +
                               std::to_string(n_total_bins) +
                               " bins with sufficient generation index bits. Need at least " +
                               std::to_string(index_bits + min_gen_bits) + " bits, but only " +
                               std::to_string(total_bits) + " are available.");
    }

    shift_amount_ = static_cast<S>(index_bits);
    // Create mask: (1 << index_bits) - 1.
    index_mask_ = static_cast<S>((S(1) << index_bits) - 1);

    // Max generation: remaining bits.
    const unsigned int gen_bits = total_bits - index_bits;
    max_gen_ = static_cast<S>((S(1) << gen_bits) - 1);

    // Initialize map with 0
    sparse_map_.assign(static_cast<std::size_t>(n_total_bins), S(0));

    // Heuristic reserve
    reserve_size = std::min(static_cast<std::size_t>(n_total_bins), reserve_size);
    dense_ids_.reserve(reserve_size);
    dense_acc_.reserve(reserve_size);

    current_gen_ = 1;
  }

  /**
   * @brief Accumulates a weight into a global bin index.
   *
   * This is the performance-critical hot path function. It updates the thread-local
   * buffer using a sparse-set inspired O(1) validity check. Weights are summed
   * linearly within an event to ensure correct error propagation during flushes.
   *
   * @param global_idx The flattened global index of the bin in `HistogramData`.
   * @param w The weight to accumulate.
   *
   * @note If the bin was already touched in the current event (generation), the
   *       weight is added to the existing accumulator. Otherwise, a new entry
   *       is created in the dense buffer.
   */
  inline void fill(S global_idx, const T& w) noexcept {
    // 1. Read the packed value (Generation | BufferIndex)
    const S packed = sparse_map_[static_cast<std::size_t>(global_idx)];

    // 2. Validity Check
    // If the high bits match current_gen_, the lower bits are a valid index
    if ((packed >> shift_amount_) == current_gen_) [[likely]] {
      const S idx = packed & index_mask_;
      dense_acc_[static_cast<std::size_t>(idx)].add(w);
    } else {
      // 3. Miss: New entry for this event
      const S new_idx = static_cast<S>(dense_ids_.size());

      // Update map: Store (CurrentGen << Shift) | NewIndex
      sparse_map_[static_cast<std::size_t>(global_idx)] = (current_gen_ << shift_amount_) | new_idx;

      // Track this bin for flushing
      dense_ids_.push_back(global_idx);

      // Initialize accumulator
      dense_acc_.emplace_back(w);
    }
  }

  /**
   * @brief Flushes the buffered results to the global storage.
   *
   * This must be called at the end of each event (or whenever event-level
   * consolidation is required). It computes the net weight for each bin touched
   * in the event and adds it to the global storage.
   *
   * The variance contribution is calculated as the square of the net weight
   * accumulated within the event, which is the standard procedure for handling
   * correlated weights or cancellations in Monte Carlo event generation.
   *
   * @param hist_data The global histogram data storage to receive the results.
   */
  void flush(HistogramData<NT>& hist_data) {
    // Iterate only over the bins touched in this event
    for (std::size_t i = 0; i < dense_ids_.size(); ++i) {
      const S gid = dense_ids_[i];

      // Get the net weight for this bin in this event (with high precision)
      const T net_weight = dense_acc_[i].result();

      // Accumulate to global:
      // Value += net_weight
      // Error += (net_weight)^2  <-- Correct physics logic for cancellation events
      hist_data.accumulate(gid, net_weight, net_weight * net_weight);
    }

    // Increment event counter in global storage
    hist_data.increment_count();

    // Reset dense vectors (keeps capacity, size becomes 0)
    dense_ids_.clear();
    dense_acc_.clear();

    // Advance generation
    current_gen_++;

    // Check for generation overflow
    if (current_gen_ > max_gen_) {
      std::fill(sparse_map_.begin(), sparse_map_.end(), S(0));
      current_gen_ = 1;
    }
  }

 private:
  // Sparse map: GlobalBinIndex -> Packed(Generation, BufferIndex)
  std::vector<S> sparse_map_;

  // Dense arrays (Structure of Arrays) for cache efficiency
  std::vector<S> dense_ids_;
  std::vector<acc_type> dense_acc_;

  S current_gen_ = 1;
  S shift_amount_ = 0;
  S index_mask_ = 0;
  S max_gen_ = 0;
};

}  // namespace kakuhen::histogram
