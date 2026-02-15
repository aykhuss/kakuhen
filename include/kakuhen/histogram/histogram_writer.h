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

  /**
   * @brief Write the global header.
   * @tparam Registry The type of the histogram registry.
   * @param reg The histogram registry.
   */
  template <typename Registry>
  void global_header(const Registry& reg) {
    derived().global_header_impl(reg);
  }

  /**
   * @brief Write the header for a specific histogram.
   * @param i The index of the histogram.
   * @param name The name of the histogram.
   * @param nbins The total number of bins.
   * @param nvalues The number of values per bin.
   * @param ndim The number of dimensions.
   * @param ranges The bin ranges for each dimension.
   * @param neval The number of evaluations.
   */
  void histogram_header(size_type i, const std::string_view name, size_type nbins,
                        size_type nvalues, size_type ndim,
                        const std::vector<std::vector<BinRange<value_type>>>& ranges,
                        count_type neval) {
    derived().histogram_header_impl(i, name, nbins, nvalues, ndim, ranges, neval);
  }

  /**
   * @brief Write a row of data for a specific bin.
   * @param ibin The flat index of the bin.
   * @param bin_range The range of the bin in each dimension.
   * @param values The values in the bin.
   * @param errors The errors in the bin.
   */
  void histogram_row(size_type ibin, const std::vector<BinRange<value_type>>& bin_range,
                     const std::vector<value_type>& values, const std::vector<value_type>& errors) {
    derived().histogram_row_impl(ibin, bin_range, values, errors);
  }

  /**
   * @brief Write the footer for a specific histogram.
   */
  void histogram_footer() {
    derived().histogram_footer_impl();
  }

  /**
   * @brief Write the global footer.
   */
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

/**
 * @brief Writer implementation for the NNLOJET format.
 *
 * This writer outputs histograms in a format compatible with NNLOJET analysis tools.
 * It currently supports only 1D histograms.
 *
 * @tparam NT The numeric traits to use.
 */
template <typename NT = util::num_traits_t<>>
class NNLOJETWriter : public HistogramWriter<NNLOJETWriter<NT>, NT> {
 public:
  using Base = HistogramWriter<NNLOJETWriter<NT>, NT>;
  // shorthands to save typing
  using S = typename Base::size_type;
  using T = typename Base::value_type;
  using U = typename Base::count_type;

  /**
   * @brief Construct a new NNLOJETWriter.
   * @param os The output stream to write to.
   */
  explicit NNLOJETWriter(std::ostream& os) : Base(os), neval_{0} {}

  /**
   * @brief Reset the writer state.
   */
  void reset() {
    Base::reset();
    neval_ = 0;
  }

  /**
   * @brief Implementation of global_header.
   */
  template <typename Registry>
  void global_header_impl([[maybe_unused]] const Registry& reg) {}

  /**
   * @brief Implementation of histogram_header.
   *
   * Writes the NNLOJET-specific header including name, labels, and number of evaluations.
   */
  void histogram_header_impl([[maybe_unused]] S i, const std::string_view name,
                             [[maybe_unused]] S nbins, S nvalues, S ndim,
                             [[maybe_unused]] const std::vector<std::vector<BinRange<T>>>& ranges,
                             U neval) {
    assert(ndim == 1 && "NNLOJET only support 1D histograms");
    os_ << std::format("#name: {}\n", name);
    os_ << std::format("#labels: {0}_lower[1]   {0}_center[2]   {0}_upper[3] ", name);
    for (S ival = 0; ival < nvalues; ++ival) {
      os_ << std::format(" value{0}[{1}] error{0}[{2}] ", ival + 1, ival + 4, ival + 5);
    }
    os_ << std::format("\n");
    os_ << std::format("#neval: {}\n", neval);
  }

  /**
   * @brief Implementation of histogram_row.
   *
   * Writes the bin edges, centers, values, and errors.
   * Normalizes the values by the bin width (Jacobian).
   */
  void histogram_row_impl([[maybe_unused]] S ibin, const std::vector<BinRange<T>>& bin_range,
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
    for (std::size_t i = 0; i < values.size(); ++i) {
      os_ << std::format(" {:.16e} {:.16e} ", jac * values[i], jac * errors[i]);
    }
    os_ << std::format("\n");
  }

  /**
   * @brief Implementation of histogram_footer.
   */
  void histogram_footer_impl() {
    os_ << std::format("#nx: {}\n\n", 3);
  }

  /**
   * @brief Implementation of global_footer.
   */
  void global_footer_impl() {}

 private:
  friend class HistogramWriter<NNLOJETWriter<NT>, NT>;
  using Base::os_;
  U neval_;
};

}  // namespace kakuhen::histogram
