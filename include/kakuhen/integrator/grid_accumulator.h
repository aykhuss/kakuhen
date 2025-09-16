#pragma once

#include "kakuhen/util/serialize.h"
#include "kakuhen/util/type.h"

namespace kakuhen::integrator {

template <typename T, typename U>
struct GridAccumulator {
  using value_type = T;
  using count_type = U;

  T acc_ = T(0);
  U n_ = U(0);

  //> methods to manipulate state

  inline void accumulate(const T& x) noexcept {
    acc_ += x;
    n_++;
  }

  inline void accumulate(const GridAccumulator<T, U>& other) noexcept {
    acc_ += other.acc_;
    n_ += other.n_;
  }

  inline void reset() noexcept {
    acc_ = T(0);
    n_ = U(0);
  }

  inline void reset(const T& acc, const U& n) noexcept {
    acc_ = acc;
    n_ = n;
  }

  //> query methods

  // number of calls
  inline U count() const noexcept {
    return n_;
  }

  // value of the accumulator
  inline T value() const noexcept {
    return acc_;
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
    kakuhen::util::serialize::serialize_one<T>(out, acc_);
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
    kakuhen::util::serialize::deserialize_one<T>(in, acc_);
    kakuhen::util::serialize::deserialize_one<U>(in, n_);
  }

};  // struct GridAccumulator

}  // namespace kakuhen::integrator
