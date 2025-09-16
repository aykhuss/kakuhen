#pragma once

// note: this implementation is not thread safe!

#include "kakuhen/util/serialize.h"
#include "kakuhen/util/type.h"
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace kakuhen::util::accumulator {

enum class AccumAlgo { Naive, Kahan, Neumaier, TwoSum };

namespace detail {

//> TwoSum algorithm for summation of two numbers a,b
//> where the sum s = (a+b) is rounded to the nearest
//> floating point number. The algorithm futher computes
//> the error t of the summation, such that
//>     a+b = s+t
//> *exactly*. Compared to the original "FastTwoSum"
//> algorithm by [Dekker] and [Kahan], this one
//> requires no comparison to determine the hierarchy
//> between a,b and is thus quicker, moreover, it works
//> with any radix
//> Algorithm by [Knuth] and [Moller].
//> more info here:
//> https://people.eecs.berkeley.edu/~jrs/papers/robust-predicates.pdf
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

//> CRTP base class
template <typename Derived, typename T>
class AccumulatorBase {
  static_assert(std::is_floating_point_v<T>, "Accumulator only supports floating point types");

 public:
  constexpr void add(const T& value) noexcept {
    derived().add_impl(value);
  }
  constexpr T result() const noexcept {
    return derived().result_impl();
  }
  constexpr void reset(const T& value = T(0)) noexcept {
    derived().reset_impl(value);
  }

  //> convenient operators
  constexpr Derived& operator+=(const T& value) noexcept {
    derived().add_impl(value);
    return derived();
  }
  constexpr operator T() const noexcept {
    return derived().result_impl();
  }

  //> this on its own is broken for derived classes
  //> need to implement/delete copy/move constructors
  // constexpr Derived& operator=(T value) noexcept {
  //   derived().reset_impl(value);
  //   return *static_cast<Derived*>(this);
  // }

  //> serialization support

  void serialize(std::ostream& out, bool with_type = false) const noexcept {
    if (with_type) {
      int16_t T_tos = kakuhen::util::type::get_type_or_size<T>();
      kakuhen::util::serialize::serialize_one<int16_t>(out, T_tos);
    }
    kakuhen::util::serialize::serialize_one<T>(out, result());
  }

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

 protected:
  inline Derived& derived() {
    return static_cast<Derived&>(*this);
  }

  inline const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }
};

//> Naive
template <typename T>
class NaiveAccumulator final : public AccumulatorBase<NaiveAccumulator<T>, T> {
 public:
  constexpr explicit NaiveAccumulator(const T& initial = T(0)) noexcept : sum_(initial) {}

  constexpr void add_impl(const T& value) noexcept {
    sum_ += value;
  }
  constexpr T result_impl() const noexcept {
    return sum_;
  }
  constexpr void reset_impl(const T& value = T(0)) noexcept {
    sum_ = value;
  }

 private:
  T sum_ = T(0);
};

//> Kahan
template <typename T>
class KahanAccumulator final : public AccumulatorBase<KahanAccumulator<T>, T> {
 public:
  constexpr explicit KahanAccumulator(const T& initial = T(0)) noexcept : sum_(initial), c_(T(0)) {}

  constexpr void add_impl(const T& value) noexcept {
    T y = value - c_;
    T t = sum_ + y;
    c_ = (t - sum_) - y;
    sum_ = t;
  }
  constexpr T result_impl() const noexcept {
    return sum_;
  }
  constexpr void reset_impl(const T& value = T(0)) noexcept {
    sum_ = value;
    c_ = T(0);
  }

 private:
  T sum_ = T(0);
  T c_ = T(0);
};

//> Neumaier
template <typename T>
class NeumaierAccumulator final : public AccumulatorBase<NeumaierAccumulator<T>, T> {
 public:
  constexpr explicit NeumaierAccumulator(const T& initial = T(0)) noexcept
      : sum_(initial), c_(T(0)) {}

  constexpr void add_impl(const T& value) noexcept {
    T t = sum_ + value;
    if (std::fabs(sum_) >= std::fabs(value))
      c_ += (sum_ - t) + value;
    else
      c_ += (value - t) + sum_;
    sum_ = t;
  }
  constexpr T result_impl() const noexcept {
    return sum_ + c_;
  }
  constexpr void reset_impl(const T& value = T(0)) noexcept {
    sum_ = value;
    c_ = T(0);
  }

 private:
  T sum_ = T(0);
  T c_ = T(0);
};

//> TwoSum
template <typename T>
class TwoSumAccumulator final : public AccumulatorBase<TwoSumAccumulator<T>, T> {
 public:
  static inline constexpr T epsilon = std::numeric_limits<T>::epsilon() * 10;

  constexpr explicit TwoSumAccumulator(const T& initial = T(0)) noexcept
      : sum_(initial), c_(T(0)) {}

  constexpr void add_impl(const T& value) noexcept {
    T sum, err;
    detail::two_sum<T>(sum_, value, sum, err);
    sum_ = sum;
    c_ += err;  // accumulate low-order bits
    //> keep c_ small by renormalizing occasionally
    if (std::fabs(c_) > epsilon * std::fabs(sum_)) {
      renorm();
    }
  }

  //> alternative implementation of add_impl (less accurate but faster)
  // constexpr void add_impl(const T& value) noexcept {
  //   T s = sum_ + value;
  //   T bp = s - sum_;
  //   T err = (sum_ - (s - bp)) + (value - bp);
  //   sum_ = s;
  //   c_ += err;
  // }

  constexpr T result_impl() const noexcept {
    return sum_ + c_;
  }
  constexpr void reset_impl(const T& value = T(0)) noexcept {
    sum_ = value;
    c_ = T(0);
  }

 private:
  constexpr void renorm() noexcept {
    T s, e;
    detail::two_sum<T>(sum_, c_, s, e);
    sum_ = s;
    c_ = e;
  }
  T sum_ = T(0);
  T c_ = T(0);
};

//> Enum →  Type mapping
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

//> Alias
template <AccumAlgo Algo, typename T = double>
using Accumulator = typename detail::AccumulatorSelector<Algo, T>::type;

}  // namespace kakuhen::util::accumulator
