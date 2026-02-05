#pragma once

#include "kakuhen/histogram/axis.h"
#include "kakuhen/histogram/bin_range.h"
#include "kakuhen/histogram/histogram_registry.h"
#include "kakuhen/kakuhen.h"
#include "kakuhen/ndarray/ndarray.h"
#include "kakuhen/util/accumulator.h"
#include "kakuhen/util/math.h"
#include "kakuhen/util/numeric_traits.h"
#include "kakuhen/util/printer.h"
#include <algorithm>
#include <argparse/argparse.hpp>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <stack>
#include <sstream>
#include <string>

/**
 * @brief Printer that emits gnuplot-friendly output from integrator state.
 *
 * This printer interprets the JSON-like printer callbacks used by kakuhen
 * and converts them into gnuplot data blocks and variables.
 */
class GnuplotPrinter : public kakuhen::util::printer::PrinterBase<GnuplotPrinter> {
 public:
  /**
   * @brief Internal printer state used to route data to gnuplot blocks.
   */
  enum class GnuplotContext : uint8_t { GRID1D, GRID2D, DIMS, GRID, ORDER, OTHER };
  /**
   * @brief Convert a gnuplot context to a readable string.
   */
  [[nodiscard]] static constexpr std::string_view to_string(GnuplotContext ctx) noexcept {
    switch (ctx) {
      case GnuplotContext::GRID1D:
        return "GRID1D";
      case GnuplotContext::GRID2D:
        return "GRID2D";
      case GnuplotContext::DIMS:
        return "DIMS";
      case GnuplotContext::GRID:
        return "GRID";
      case GnuplotContext::ORDER:
        return "ORDER";
      case GnuplotContext::OTHER:
        return "OTHER";
    }
    return "Unknown";
  }

  // dependent class: need to explicitly load things from the Base
  using Base = kakuhen::util::printer::PrinterBase<GnuplotPrinter>;

  /**
   * @brief Construct a gnuplot printer writing to a stream.
   * @param os Output stream to receive gnuplot blocks.
   */
  explicit GnuplotPrinter(std::ostream& os) : Base(os) {
    reset();
  }

  /*!
   * @brief Resets the Gnuplot printer to its initial state.
   */
  void reset() {
    Base::reset();
    context_stack_ = {};
    first_ = true;
    stage_ = 0;
  }

  /**
   * @brief Line break hook (no-op for this printer).
   */
  void break_line() {}

  /*!
   * @brief Begins a new Gnuplot object or array.
   *
   * @tparam C The context type (Context::OBJECT or Context::ARRAY).
   * @param key An optional key for the object member or array.
   */
  template <kakuhen::util::printer::Context C>
  void begin(std::string_view key = {}) {
    first_ = true;
    /// GRID1D, GRID2D, DIMS, GRID, ORDER, OTHER
    if (key == "grid1d") {
      context_stack_.push(GnuplotContext::GRID1D);
      stage_ = 1;
    } else if (key == "grid2d") {
      context_stack_.push(GnuplotContext::GRID2D);
      stage_ = 2;
    } else if (key == "dims") {
      context_stack_.push(GnuplotContext::DIMS);
      os_ << "\n$GRID_";
      sep_ = '_';
    } else if (key == "grid") {
      context_stack_.push(GnuplotContext::GRID);
      if (stage_ == 1) {
        sep_ = '\n';
      } else if (stage_ == 2) {
        sep_ = ' ';
      } else {
        assert(false && "Unexpected grid stage");
      }
    } else if (key == "order") {
      os_ << "\n$ORDER << EOD\n";
      context_stack_.push(GnuplotContext::ORDER);
      stage_ = 3;
    } else {
      context_stack_.push(GnuplotContext::OTHER);
      first_ = false;
      // sep_ = '#';
    }
    // os_ << "<" << to_string(context_stack_.top()) << ">";
    if (stage_ == 0) return;
    prefix(key);
  }

  /*!
   * @brief Ends the current Gnuplot object or array.
   *
   * @tparam C The context type.
   * @param do_break Whether to force a line break after ending the context.
   */
  template <kakuhen::util::printer::Context C>
  void end(bool do_break = false) {
    /// GRID1D, GRID2D, DIMS, GRID, ORDER, OTHER
    // os_ << "</" << to_string(context_stack_.top()) << ">";
    GnuplotContext context_exit = context_stack_.top();
    context_stack_.pop();
    first_ = false;

    if (C == kakuhen::util::printer::Context::ARRAY && !context_stack_.empty() &&
        (context_stack_.top() == GnuplotContext::GRID ||
         context_stack_.top() == GnuplotContext::ORDER)) {
      os_ << "\n";
    }
    if (context_exit == GnuplotContext::DIMS) {
      os_ << " << EOD\n";
    }
    if ((context_exit == GnuplotContext::GRID) || (context_exit == GnuplotContext::ORDER)) {
      os_ << "\nEOD\n";
    }

    if ((context_exit == GnuplotContext::GRID1D) || (context_exit == GnuplotContext::GRID2D) ||
        (context_exit == GnuplotContext::ORDER)) {
      stage_ = 0;
    }

    if (stage_ == 0) return;
  }

 private:
  friend class PrinterBase<GnuplotPrinter>;

  using Base::os_;
  uint8_t stage_ = 0;  // 0: idle, 1: grid1d, 2: grid2d, 3: order
  std::stack<GnuplotContext> context_stack_;
  bool first_;
  char sep_ = ' ';

  template <typename T>
  void print_one_impl(std::string_view key, T&& value) {
    if (key.starts_with("ndiv")) {
      os_ << std::format("{} = {};\n", key, value);
    }
    if (stage_ == 0) return;
    prefix(key);
    if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
      /// the `+` trick here forces that uint8_t are not printed as char
      os_ << +std::forward<T>(value);
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::string_view> ||
                         std::is_same_v<std::decay_t<T>, std::string> ||
                         std::is_constructible_v<std::string_view, T>) {
      os_ << std::forward<T>(value);
    } else {
      static_assert(sizeof(T) == 0, "Unsupported type for print_one");
    }
    first_ = false;
  }

  void prefix(std::string_view key) {
    if (!first_) os_ << sep_;
    // os_ << sep_;
  }
};

template <typename NT = kakuhen::util::num_traits_t<>>
/**
 * @brief Helper integrand to collect histogram samples and emit gnuplot plots.
 *
 * Builds 1D and 2D histograms for each dimension and prints gnuplot commands
 * to visualize the sampling distribution.
 */
struct GnuplotSample {
  using num_traits = NT;
  using UniformAxis_t =
      kakuhen::histogram::UniformAxis<typename num_traits::value_type, typename num_traits::size_type>;
  using VariableAxis_t =
      kakuhen::histogram::VariableAxis<typename num_traits::value_type, typename num_traits::size_type>;
  using T = num_traits::value_type;
  using S = num_traits::size_type;
  using U = num_traits::count_type;
  using registry_type = kakuhen::histogram::HistogramRegistry<num_traits>;
  using buffer_type =
      kakuhen::histogram::HistogramBuffer<num_traits, kakuhen::util::accumulator::TwoSumAccumulator<T>>;
  using id_type = registry_type::Id;

  /**
   * @brief Construct a sampler with uniform 1D and 2D histograms.
   * @param ndim Number of dimensions.
   * @param ndiv Grid divisions per dimension.
   */
  GnuplotSample(S ndim, S ndiv)
      : ndim_{ndim}, ndiv_{ndiv}, registry_{}, buffer_{}, ids_({ndim, ndim}) {
    UniformAxis_t uni_1D(2 * ndiv, 0, 1);
    UniformAxis_t uni_2D(ndiv, 0, 1);
    for (S idim = 0; idim < ndim; ++idim) {
      ids_(idim, idim) = registry_.book(std::to_string(idim), 1, uni_1D);
      for (S jdim = 0; jdim < idim; ++jdim) {
        ids_(idim, jdim) =
            registry_.book(std::to_string(idim) + "_" + std::to_string(jdim), 1, uni_2D, uni_2D);
      }
    }
    buffer_ = registry_.create_buffer();
  }

  /**
   * @brief Accumulate one point into all relevant histograms.
   * @param point Sample point from the integrator.
   * @return Constant weight (1).
   */
  T operator()(const kakuhen::integrator::Point<num_traits>& point) {
    assert(point.ndim == ndim_);
    const auto& x = point.x;

    for (S idim = 0; idim < ndim_; ++idim) {
      registry_.fill(buffer_, ids_(idim, idim), T(1), x[idim]);
      for (S jdim = 0; jdim < idim; ++jdim) {
        registry_.fill(buffer_, ids_(idim, jdim), T(1), x[idim], x[jdim]);
      }
    }
    registry_.flush(buffer_);

    return T(1);
  }

  /**
   * @brief Print gnuplot data blocks and plotting commands to stdout.
   */
  void print(std::ostream& out) {
    /// registry
    std::ostringstream record;
    T jac;
    for (const auto& id : registry_.ids()) {
      const auto nbins = registry_.get_nbins(id);
      const auto nvalues = registry_.get_nvalues(id);
      const auto ndim = registry_.get_ndim(id);
      const auto ranges = registry_.get_bin_ranges(id);
      assert(ranges.size() == static_cast<std::size_t>(ndim));
      out << std::format("\n\n$DATA_{} << EOD\n", registry_.get_name(id));
      for (S idx_flat = 0; idx_flat < nbins; ++idx_flat) {
        const auto idx_bins = registry_.get_bin_indices(id, idx_flat);
        assert(idx_bins.size() == static_cast<std::size_t>(ndim));
        bool skip_record = false;
        record.str("");
        record.clear();
        jac = 1;
        // record << std::format("{:>7d}  ", idx_flat);
        if (idx_bins[ndim - 1] == 0) out << "\n";
        for (S idim = 0; idim < ndim; ++idim) {
          const S iidx = idx_bins[idim];
          const auto& irange = ranges[idim][iidx];
          if (irange.kind == kakuhen::histogram::BinKind::Underflow ||
              irange.kind == kakuhen::histogram::BinKind::Invalid) {
            skip_record = true;
            break;
          }
          record << std::format(" {:16.8G} {:16.8G} ", irange.low, irange.upp);
          jac /= (irange.upp - irange.low);
        }  // for idim
        if (skip_record) continue;
        record << "  ";
        for (S ival = 0; ival < nvalues; ++ival) {
          const auto& val = jac * registry_.get_bin_value(id, idx_flat, ival);
          const auto& err = jac * registry_.get_bin_error(id, idx_flat, ival);
          record << std::format(" {:16.8G} {:16.8G} ", val, err);
        }
        out << record.str() << "\n";
      }  // for idx_flat
      out << "EOD\n";
    }  // for id

    out << std::format(
        "\n\n"
        "set terminal pdfcairo enhanced color transparent dashed "
        "size {0}cm,{0}cm font \"Iosevka Bold\" fontscale 1.0\n",
        5. * ndim_);

    out << R"(
set encoding utf8

set view map
set pm3d map explicit noborder corners2color c1  # bottom left corner
# set pm3d interpolate 2,2
# set palette rgb 33,13,10;
set palette defined ( 0 'dark-blue', 1/8. 'blue', 3/8. 'cyan', 5/8. 'yellow', 7/8. 'red', 1 'dark-red' )
set style rectangle front fs empty border lc rgb '#000000' lw 0.5

unset title
unset key
unset xlabel
unset ylabel
unset y2label
unset zlabel
set format x ""
set format y ""
set format y2 ""
set format z ""
unset colorbox
set format cb ""
set tics scale 0.5

if ( !exists("ndiv") && exists("ndiv0") ) {
  ndiv = ndiv0
} else {
  stats $GRID_0 using 1 nooutput
  ndiv = STATS_records - 1
}

)";

    out << "set output \"plot.pdf\"\n";
    out << std::format(
        "set multiplot layout {0},{0} "
        "margins 0,1,0,1 "
        "spacing 0.01,0.01\n"
        "set size square\n\n",
        ndim_);

    for (S idim = 0; idim < ndim_; ++idim) {
      for (S jdim = 0; jdim < ndim_; ++jdim) {
        out << "\nunset label; unset object;\n";
        out
            << R"(do for [o=1:3] { if (word($ORDER[o], 1) == )" << idim
            << R"( && word($ORDER[o], 2) == )" << jdim
            << R"() { set label 9 "".o at graph 0.9, 0.9 back center font ",10" textcolor rgb "#23d20f39" } })"
            << "\n";
        if (idim == jdim) {
          out << std::format(
              "set label 1 \"{0}\" at graph 0.6, 0.2 center back "
              "font \",30\" textcolor rgb \"#DD000000\"\n"
              "set xrange [0:1]; set yrange [0:1]; set y2range [0:*];\n"
              "plot \\\n"
              "  $DATA_{0} u 1:3 axes x1y2 with fillsteps fillstyle solid lc rgb \"#DD1e66f5\" "
              "notitle,\\\n"
              "  $DATA_{0} u 1:3 axes x1y2 with steps lw 1                lc rgb   \"#1e66f5\" "
              "notitle,\\\n"
              "  $GRID_{0} u 1:(($0+0.)/(ndiv+0.)) axes x1y1 with lines lw 2 lc rgb \"#4c4f69\" "
              "notitle\n",
              idim);
          continue;
        }
        S ix = 1;
        S iy = 3;
        if (idim > jdim) {
          std::swap(ix, iy);
        }
        out
            << R"(do for [r=1:ndiv1] { eval system("awk -v io=".(r*ndiv0)." '{for(i=3;i<NF;i++){printf(\"set object %d rect from %e,%e to %e,%e front;\",io,$(i),$1,$(i+1),$2);io++}}' <<< '" .)"
            << std::format("$GRID_{}_{}", idim, jdim) << R"([r] . "'") })" << "\n";
        out << std::format(
            "set label 1 \"{0}\" at graph 0.6, 0.2 center back "
            "font \",30\" textcolor rgb \"#66000000\"\n"
            "set label 2 \"{1}\" at graph 0.2, 0.6 center back "
            "font \",30\" textcolor rgb \"#66000000\"\n"
            "set xrange [0:1]; set yrange [0:1]; set cbrange [0:*];\n"
            "splot $DATA_{2}_{3} u {4}:{5}:5 "
            "with pm3d lc palette z fs transparent solid 0.75 notitle\n",
            jdim, idim, kakuhen::util::math::max(idim, jdim),
            kakuhen::util::math::min(idim, jdim), ix, iy);
      }
    }

    out << "unset multiplot\n";
  }

  S ndim_;
  S ndiv_;
  registry_type registry_;
  buffer_type buffer_;
  kakuhen::ndarray::NDArray<id_type, S> ids_;
};
