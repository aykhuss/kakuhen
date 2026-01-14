#pragma once

/// @note This implementation is not thread-safe.

#include "kakuhen/util/math.h"
#include "kakuhen/util/serialize.h"
#include "kakuhen/util/type.h"
#include <cstddef>
#include <type_traits>

namespace kakuhen::util::accumulator {

/// @brief Enum defining the available accumulation algorithms.
enum class AccumAlgo {
  Naive,     //!< Straightforward summation.
  Kahan,     //!< Kahan compensated summation.
  Neumaier,  //!< Neumaier's improved Kahan summation.
  TwoSum     //!< High-precision Two-Sum algorithm.
};

namespace detail {

/*!
 * @brief Two-Sum algorithm for exact summation of two numbers.
 *
 * This function computes the sum \f$ s = a + b \f$ rounded to the nearest
 * floating-point number, and the exact error \f$ t \f$ such that
 * \f$ a + b = s + t \f$ holds exactly in floating-point arithmetic.
 *
 * Compared to the original "FastTwoSum" algorithm by Dekker and Kahan, this
 * version (by Knuth and Møller) requires no comparisons to determine the
 * relative magnitude of \f$ a \f$ and \f$ b \f$, making it faster on modern
 * architectures. It works with any radix.
 *
 * @tparam T The floating-point type (defaults to double).
 * @param a First number to add.
 * @param b Second number to add.
 * @param s Output parameter: the rounded sum \f$ a + b \f$.
 * @param t Output parameter: the exact rounding error.
 *
 * @see https://people.eecs.berkeley.edu/~jrs/papers/robust-predicates.pdf
 */
template <typename T = double>
constexpr void two_sum(const T& a, const T& b, T& s, T& t) noexcept {
  s = a + b;
  T ap = s - b;
  T bp = s - ap;  // yes, this is supposed to be `ap`!
  T da = a - ap;
  T db = b - bp;
  t = da + db;
}

}  // namespace detail

/*!
 * @brief Base class for various floating-point summation algorithms using CRTP.
 *
 * This class provides a common interface for different accumulation strategies
 * to sum floating-point numbers with varying degrees of precision.
 *
 * @tparam Derived The derived accumulator class.
 * @tparam T The floating-point type to accumulate.
 */
template <typename Derived, typename T>
class AccumulatorBase {
  static_assert(std::is_floating_point_v<T>, "Accumulator only supports floating point types");

 public:
  /*!
   * @brief Adds a value to the accumulator.
   * @param value The value to add.
   */
  constexpr void add(const T& value) noexcept {
    derived().add_impl(value);
  }

  /*!
   * @brief Gets the current accumulated result.
   * @return The accumulated sum.
   */
  [[nodiscard]] constexpr T result() const noexcept {
    return derived().result_impl();
  }

  /*!
   * @brief Resets the accumulator to an initial value.
   * @param value The value to reset the accumulator to. Defaults to 0.
   */
  constexpr void reset(const T& value = T(0)) noexcept {
    derived().reset_impl(value);
  }

  /// @name Operators
  /// @{

  /*!
   * @brief Overloads the += operator to add a value.
   * @param value The value to add.
   * @return A reference to the derived object.
   */
  constexpr Derived& operator+=(const T& value) noexcept {
    derived().add_impl(value);
    return derived();
  }

  // // feature dropped: would need to explicitly load it in each derived class
  // // and user should be more expllicit and just use `reset` anyway
  // /*!
  //  * @brief Assignment operator to reset the accumulator to a value.
  //  * @param value The value to reset to.
  //  * @return A reference to the derived object.
  //  */
  // constexpr Derived& operator=(T value) noexcept {
  //   derived().reset_impl(value);
  //   return derived();
  // }

  /*!
   * @brief Conversion operator to allow the accumulator to be used as its underlying value type.
   */
  [[nodiscard]] constexpr operator T() const noexcept {
    return derived().result_impl();
  }
  /// @}

  /// @name Serialization
  /// @{

  /*!
   * @brief Serializes the accumulator to an output stream.
   * @param out The output stream.
   * @param with_type Whether to include type information.
   */
  void serialize(std::ostream& out, bool with_type = false) const noexcept {
    if (with_type) {
      int16_t T_tos = kakuhen::util::type::get_type_or_size<T>();
      kakuhen::util::serialize::serialize_one<int16_t>(out, T_tos);
    }
    kakuhen::util::serialize::serialize_one<T>(out, result());
  }

  /*!
   * @brief Deserializes the accumulator from an input stream.
   * @param in The input stream.
   * @param with_type Whether to verify type information.
   */
  void deserialize(std::istream& in, bool with_type = false) {
    if (with_type) {
      int16_t T_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, T_tos);
      if (T_tos != kakuhen::util::type::get_type_or_size<T>()) {
        throw std::runtime_error("type or size mismatch for typename T");
      }
    }
    T value;
    kakuhen::util::serialize::deserialize_one<T>(in, value);
    reset(value);
  }
  /// @}

 protected:
  inline Derived& derived() {
    return static_cast<Derived&>(*this);
  }

  inline const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }
};

/*!
 * @brief Implements naive floating-point summation.
 *
 * This accumulator performs straightforward summation (`sum_ += value`).
 * It is the simplest and fastest method but can suffer from significant
 * precision loss.
 *
 * @tparam T The floating-point type.
 */
template <typename T>
class NaiveAccumulator final : public AccumulatorBase<NaiveAccumulator<T>, T> {
 public:
  constexpr explicit NaiveAccumulator(const T& initial = T(0)) noexcept : sum_(initial) {}

 private:
  friend class AccumulatorBase<NaiveAccumulator<T>, T>;

  constexpr void add_impl(const T& value) noexcept {
    sum_ += value;
  }
  [[nodiscard]] constexpr T result_impl() const noexcept {
    return sum_;
  }
  constexpr void reset_impl(const T& value = T(0)) noexcept {
    sum_ = value;
  }

  T sum_ = T(0);
};

/*!
 * @brief Implements Kahan summation algorithm for improved precision.
 *
 * The Kahan summation algorithm (compensated summation) keeps track of a
 * running compensation for lost low-order bits.
 *
 * @tparam T The floating-point type.
 */
template <typename T>
class KahanAccumulator final : public AccumulatorBase<KahanAccumulator<T>, T> {
 public:
  constexpr explicit KahanAccumulator(const T& initial = T(0)) noexcept : sum_(initial), c_(T(0)) {}

 private:
  friend class AccumulatorBase<KahanAccumulator<T>, T>;

  constexpr void add_impl(const T& value) noexcept {
    T y = value - c_;
    T t = sum_ + y;
    c_ = (t - sum_) - y;
    sum_ = t;
  }
  [[nodiscard]] constexpr T result_impl() const noexcept {
    return sum_;
  }
  constexpr void reset_impl(const T& value = T(0)) noexcept {
    sum_ = value;
    c_ = T(0);
  }

  T sum_ = T(0);
  T c_ = T(0);
};

/*!
 * @brief Implements Neumaier's improved Kahan summation algorithm.
 *
 * Neumaier's algorithm improves upon Kahan summation for cases where the
 * added term is larger than the running sum.
 *
 * @tparam T The floating-point type.
 */
template <typename T>
class NeumaierAccumulator final : public AccumulatorBase<NeumaierAccumulator<T>, T> {
 public:
  constexpr explicit NeumaierAccumulator(const T& initial = T(0)) noexcept
      : sum_(initial), c_(T(0)) {}

 private:
  friend class AccumulatorBase<NeumaierAccumulator<T>, T>;

  constexpr void add_impl(const T& value) noexcept {
    T t = sum_ + value;
    if (kakuhen::util::math::abs(sum_) >= kakuhen::util::math::abs(value))
      c_ += (sum_ - t) + value;
    else
      c_ += (value - t) + sum_;
    sum_ = t;
  }
  [[nodiscard]] constexpr T result_impl() const noexcept {
    return sum_ + c_;
  }
  constexpr void reset_impl(const T& value = T(0)) noexcept {
    sum_ = value;
    c_ = T(0);
  }

  T sum_ = T(0);
  T c_ = T(0);
};

/*!
 * @brief Implements the Two-Sum algorithm for high-precision summation.
 *
 * The Two-Sum algorithm calculates an exact error term for every addition.
 * This class separates the fast path addition from the renormalization
 * step for maximum performance.
 *
 * @tparam T The floating-point type.
 */
template <typename T>
class TwoSumAccumulator final : public AccumulatorBase<TwoSumAccumulator<T>, T> {
 public:
  // static inline constexpr T epsilon = std::numeric_limits<T>::epsilon() * 10;

  constexpr explicit TwoSumAccumulator(const T& initial = T(0)) noexcept
      : sum_(initial), c_(T(0)) {}

  // // not used for the moment
  // /*!
  //  * @brief Explicitly realigns the sum and error terms.
  //  *
  //  * This method ensures the internal error term `c_` is minimized relative to `sum_`.
  //  * It is useful after long summation sequences.
  //  */
  // constexpr void renormalize() noexcept {
  //   T s, e;
  //   detail::two_sum<T>(sum_, c_, s, e);
  //   sum_ = s;
  //   c_ = e;
  // }

 private:
  friend class AccumulatorBase<TwoSumAccumulator<T>, T>;

  constexpr void add_impl(const T& value) noexcept {
    T sum, err;
    detail::two_sum<T>(sum_, value, sum, err);
    sum_ = sum;
    c_ += err;
    // // this is bad for the branch predictor for we remove the renormalization step!
    // // instead, we provide a renormalize routine, that the user can call if need be
    // // keep c_ small by renormalizing occasionally
    // if (std::fabs(c_) > epsilon * std::fabs(sum_)) {
    //   renormalize();
    // }
  }

  [[nodiscard]] constexpr T result_impl() const noexcept {
    T s, e;
    // Perform a non-destructive renormalization to return the most accurate result
    detail::two_sum<T>(sum_, c_, s, e);
    return s + e;
  }

  constexpr void reset_impl(const T& value = T(0)) noexcept {
    sum_ = value;
    c_ = T(0);
  }

  T sum_ = T(0);
  T c_ = T(0);
};

// Enum -> Type mapping
namespace detail {

template <AccumAlgo Algo, typename T>
struct AccumulatorSelector;

template <typename T>
struct AccumulatorSelector<AccumAlgo::Naive, T> {
  using type = NaiveAccumulator<T>;
};
template <typename T>
struct AccumulatorSelector<AccumAlgo::Kahan, T> {
  using type = KahanAccumulator<T>;
};
template <typename T>
struct AccumulatorSelector<AccumAlgo::Neumaier, T> {
  using type = NeumaierAccumulator<T>;
};
template <typename T>
struct AccumulatorSelector<AccumAlgo::TwoSum, T> {
  using type = TwoSumAccumulator<T>;
};

}  // namespace detail

/*!
 * @brief Alias for selecting a specific accumulator implementation.
 * @tparam T The floating-point type. Defaults to double.
 * @tparam Algo The accumulation algorithm. Defaults to AccumAlgo::TwoSum.
 */
template <typename T = double, AccumAlgo Algo = AccumAlgo::TwoSum>
using Accumulator = typename detail::AccumulatorSelector<Algo, T>::type;

}  // namespace kakuhen::util::accumulator
