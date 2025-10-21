#pragma once

#include "kakuhen/util/accumulator.h"

namespace kakuhen::integrator {

//@todo:  update with algo in https://arxiv.org/pdf/2206.10662
// also write an accum function add(f) and add(f,f2)

template <typename T, typename U>
struct IntegralAccumulator {
  using value_type = T;
  using count_type = U;

  using acc_type =
      kakuhen::util::accumulator::Accumulator<kakuhen::util::accumulator::AccumAlgo::TwoSum, T>;

  acc_type f_{};
  acc_type f2_{};
  U n_ = 0;

  //> methods to manipulate state

  inline void accumulate(const T& f) noexcept {
    f_.add(f);
    f2_.add(f * f);
    n_++;
  }

  inline void accumulate(const T& f, const T& f2) noexcept {
    f_.add(f);
    f2_.add(f2);
    n_++;
  }

  inline void accumulate(const IntegralAccumulator<T, U>& other) noexcept {
    f_.add(other.f_.result());
    f2_.add(other.f2_.result());
    n_ += other.n_;
  }

  inline void reset() noexcept {
    f_.reset();
    f2_.reset();
    n_ = 0;
  }

  inline void reset(const T& f, const T& f2, const U& n) noexcept {
    f_.reset(f);
    f2_.reset(f2);
    n_ = n;
  }

  //> helper functions for statistics

  // number of calls
  inline U count() const noexcept {
    return n_;
  }

  // expectation value
  inline T value() const noexcept {
    return f_.result() / T(n_);
  }

  // variance of expectation value
  inline T variance() const noexcept {
    if (n_ > 1) {
      return (f2_.result() / T(n_) - value() * value()) / T(n_ - 1);
    } else {
      return T(0);
    }
  }

  // standard deviation of the expectation value
  inline T error() const noexcept {
    return std::sqrt(variance());
  }

  // @todo: more getters?

  //> serialization support

  void serialize(std::ostream& out, bool with_type = false) const noexcept {
    if (with_type) {
      int16_t T_tos = kakuhen::util::type::get_type_or_size<T>();
      int16_t U_tos = kakuhen::util::type::get_type_or_size<U>();
      kakuhen::util::serialize::serialize_one<int16_t>(out, T_tos);
      kakuhen::util::serialize::serialize_one<int16_t>(out, U_tos);
    }
    kakuhen::util::serialize::serialize_one<T>(out, f_.result());
    kakuhen::util::serialize::serialize_one<T>(out, f2_.result());
    kakuhen::util::serialize::serialize_one<U>(out, n_);
  }

  void deserialize(std::istream& in, bool with_type = false) {
    if (with_type) {
      int16_t T_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, T_tos);
      if (T_tos != kakuhen::util::type::get_type_or_size<T>()) {
        throw std::runtime_error("type or size mismatch for typename T");
      }
      int16_t U_tos;
      kakuhen::util::serialize::deserialize_one<int16_t>(in, U_tos);
      if (U_tos != kakuhen::util::type::get_type_or_size<U>()) {
        throw std::runtime_error("type or size mismatch for typename U");
      }
    }
    T value;
    kakuhen::util::serialize::deserialize_one<T>(in, value);
    f_.reset(value);
    kakuhen::util::serialize::deserialize_one<T>(in, value);
    f2_.reset(value);
    kakuhen::util::serialize::deserialize_one<U>(in, n_);
  }

};  // struct IntegralAccumulator

// create an IntegralAccumulator
template <typename T, typename U>
inline IntegralAccumulator<T, U> make_integral_accumulator(const T& value, const T& error,
                                                           const U& n) {
  T f = value * T(n);
  T f2;
  if (n > 1) {
    f2 = T(n) * (value * value + T(n - 1) * error * error);
  } else {
    f2 = f * f;  // if n==1, variance is zero
  }
  IntegralAccumulator<T, U> acc{};
  acc.f_.reset(f);
  acc.f2_.reset(f2);
  acc.n_ = n;
  return acc;
}

}  // namespace kakuhen::integrator
