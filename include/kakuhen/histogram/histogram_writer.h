#pragma once

#include "kakuhen/histogram/bin_range.h"
#include "kakuhen/util/numeric_traits.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace kakuhen::histogram {

/**
 * @brief Base class for writers using CRTP.
 */
template <typename Derived, typename NT = util::num_traits_t<>>
class HistogramWriter {
 public:
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  using count_type = typename num_traits::count_type;

  explicit HistogramWriter(std::ostream& os) : os_(os) {}

  template <typename V>
  HistogramWriter& operator<<(const V& value) {
    os_ << value;
    return *this;
  }

  void reset() {
    os_.clear();
  }

  template <typename Registry>
  void global_header(const Registry& reg) {
    derived().global_header_impl(reg);
  }

  void histogram_header(size_type i, const std::string_view name, size_type nbins,
                        size_type nvalues, size_type ndim,
                        const std::vector<std::vector<BinRange<value_type>>>& ranges,
                        count_type neval) {
    derived().histogram_header_impl(i, name, nbins, nvalues, ndim, ranges, neval);
  }

  void histogram_row(size_type ibin, const std::vector<BinRange<value_type>>& bin_range,
                     const std::vector<value_type>& values, const std::vector<value_type>& errors) {
    derived().histogram_row_impl(ibin, bin_range, values, errors);
  }

  void histogram_footer() {
    derived().histogram_footer_impl();
  }

  void global_footer() {
    derived().global_footer_impl();
  }

 protected:
  std::ostream& os_;

  inline Derived& derived() {
    return static_cast<Derived&>(*this);
  }
  inline const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }
};

template <typename NT = util::num_traits_t<>>
class NNLOJETWriter : public HistogramWriter<NNLOJETWriter<NT>, NT> {
 public:
  using Base = HistogramWriter<NNLOJETWriter<NT>, NT>;
  // shorthands to save typing
  using S = typename Base::size_type;
  using T = typename Base::value_type;
  using U = typename Base::count_type;

  explicit NNLOJETWriter(std::ostream& os) : Base(os), neval_{0} {}

  void reset() {
    Base::reset();
    neval_ = 0;
  }

  template <typename Registry>
  void global_header_impl(const Registry& reg) {}

  void histogram_header_impl(S i, const std::string_view name, S nbins, S nvalues, S ndim,
                             const std::vector<std::vector<BinRange<T>>>& ranges, U neval) {
    assert(ndim == 1 && "NNLOJET only support 1D histograms");
    os_ << std::format("#name: {}\n", name);
    os_ << std::format("#labels: {0}_lower[1]   {0}_center[2]   {0}_upper[3] ", name);
    for (S ival = 0; ival < nvalues; ++ival) {
      os_ << std::format(" value{0}[{1}] error{0}[{2}] ", ival + 1, ival + 4, ival + 5);
    }
    os_ << std::format("\n");
    os_ << std::format("#neval: {}\n", neval);
  }

  void histogram_row_impl(S ibin, const std::vector<BinRange<T>>& bin_range,
                          const std::vector<T>& values, const std::vector<T>& errors) {
    assert(bin_range.size() == 1 && "NNLOJET only support 1D histograms");
    T jac = T(1);
    for (const auto& r : bin_range) {
      if (r.kind != BinKind::Regular) return;
      const auto low = r.low;
      const auto upp = r.upp;
      const auto mid = 0.5 * (low + upp);
      jac /= (upp - low);
      os_ << std::format("{:.16e} {:.16e} {:.16e} ", low, mid, upp);
    }
    for (auto i = 0; i < values.size(); ++i) {
      os_ << std::format(" {:.16e} {:.16e} ", jac * values[i], jac * errors[i]);
    }
    os_ << std::format("\n");
  }

  void histogram_footer_impl() {
    os_ << std::format("#nx: {}\n\n", 3);
  }

  void global_footer_impl() {}

 private:
  friend class HistogramWriter<NNLOJETWriter<NT>, NT>;
  using Base::os_;
  U neval_;
};

}  // namespace kakuhen::histogram
