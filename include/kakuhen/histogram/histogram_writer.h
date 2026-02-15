#pragma once

#include "kakuhen/histogram/bin_range.h"
#include "kakuhen/util/numeric_traits.h"
#include <cassert>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace kakuhen::histogram {

template <std::floating_point F>
inline void write_sci16(std::ostream& os, F value) {
  const auto old_flags = os.flags();
  const auto old_precision = os.precision();
  os << std::scientific << std::setprecision(16) << value;
  os.flags(old_flags);
  os.precision(old_precision);
}

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
    os_ << "#name: " << name << '\n';
    os_ << "#labels: " << name << "_lower[1]   " << name << "_center[2]   " << name << "_upper[3] ";
    for (S ival = 0; ival < nvalues; ++ival) {
      os_ << " value" << (ival + 1) << '[' << (ival + 4) << "] error" << (ival + 1) << '['
          << (ival + 5) << "] ";
    }
    os_ << '\n';
    os_ << "#neval: " << neval << '\n';
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
      write_sci16(os_, low);
      os_ << ' ';
      write_sci16(os_, mid);
      os_ << ' ';
      write_sci16(os_, upp);
      os_ << ' ';
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
      os_ << ' ';
      write_sci16(os_, jac * values[i]);
      os_ << ' ';
      write_sci16(os_, jac * errors[i]);
      os_ << ' ';
    }
    os_ << '\n';
  }

  /**
   * @brief Implementation of histogram_footer.
   */
  void histogram_footer_impl() {
    os_ << "#nx: 3\n\n";
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
