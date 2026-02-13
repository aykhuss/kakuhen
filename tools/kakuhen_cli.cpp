#include "kakuhen/histogram/axis.h"
#include "kakuhen/histogram/bin_range.h"
#include "kakuhen/histogram/histogram_registry.h"
#include "kakuhen/integrator/integrator_base.h"
#include "kakuhen/kakuhen.h"
#include "kakuhen/ndarray/ndarray.h"
#include "kakuhen/util/accumulator.h"
#include "kakuhen/util/math.h"
#include "kakuhen/util/numeric_traits.h"
#include "kakuhen/util/printer.h"
#include "plot/gnuplot.h"
#include <algorithm>
#include <argparse/argparse.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

using namespace kakuhen;
using namespace integrator;
using namespace histogram;
using namespace util::printer;
using namespace util::math;

/// default numeric traits
using num_traits = util::num_traits_t<>;
using S = typename num_traits::size_type;
using T = typename num_traits::value_type;
using U = typename num_traits::count_type;

using Plain_t = Plain<num_traits>;
using Vegas_t = Vegas<num_traits>;
using Basin_t = Basin<num_traits>;
// using IntegratorVariant = std::variant<Plain_t, Vegas_t, Basin_t>;
using IntegratorVariant = std::variant<Vegas_t, Basin_t>;

inline IntegratorVariant make_integrator(const IntegratorHeader& header) {
  switch (header.id) {
    // case IntegratorId::PLAIN:
    //   return Plain_t(1);
    case IntegratorId::VEGAS:
      return Vegas_t(1);
    case IntegratorId::BASIN:
      return Basin_t(1);
    default:
      throw std::runtime_error("Unknown IntegratorId");
  }
}

int main(int argc, char* argv[]) {
  using namespace util::type;

  argparse::ArgumentParser program("kakuhen");

  //> kakuhen dump subparser
  argparse::ArgumentParser dump_cmd("dump");
  dump_cmd.add_description("dump the information of a kakuhen state file");
  dump_cmd.add_argument("file").help("kakuhen state file").nargs(1);  // exactly one file
  dump_cmd.add_argument("-i", "--indent")
      .help("number of spaces to use for JSON indentation (0 for compact output)")
      .scan<'i', int>()   // parse as integer
      .default_value(0);  // default indent level
  program.add_subparser(dump_cmd);

  //> kakuhen plot subparser
  argparse::ArgumentParser plot_cmd("plot");
  plot_cmd.add_description("plot the kakuhen state file");
  plot_cmd.add_argument("file").help("kakuhen state file").nargs(1);  // exactly one file
  plot_cmd.add_argument("--driver")
      .default_value(std::string{"gnuplot"})
      .choices("gnuplot", "matplotlib");
  plot_cmd.add_argument("--nsamples")
      .help("number of samples to use")
      .scan<'i', int>()   // parse as integer
      .default_value(0);  // default: automatic determination
  plot_cmd.add_argument("--ndivs")
      .help("number of divisions to use")
      .scan<'i', int>()   // parse as integer
      .default_value(0);  // default: automatic determination
  program.add_subparser(plot_cmd);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  if (program.is_subcommand_used("dump")) {
    auto file = dump_cmd.get<std::string>("file");
    auto indent = static_cast<uint8_t>(dump_cmd.get<int>("indent"));
    JSONPrinter jp{std::cout, indent};
    auto vint = make_integrator(parse_header(file));
    std::visit(
        [&](auto&& intg) {
          intg.load(file);
          intg.print(jp);
          jp << "\n";
        },
        vint);
  }  // dump

  if (program.is_subcommand_used("plot")) {
    auto file = plot_cmd.get<std::string>("file");
    auto header = parse_header(file);
    auto vint = make_integrator(header);

    auto output = file.substr(0, file.find_last_of('.')) + ".pdf";

    S nsamples = static_cast<S>(plot_cmd.get<int>("nsamples"));
    S ndivs = static_cast<S>(plot_cmd.get<int>("ndivs"));
    if (ndivs == 0) {
      switch (vint.index()) {
        case 0:  // Vegas_t
          ndivs = std::get<Vegas_t>(vint).ndiv();
          break;
        case 1:  // Basin_t
          ndivs = 4 * static_cast<S>(std::sqrt(std::get<Basin_t>(vint).ndiv0()));
          break;
        default:
          if (ndivs == 0) ndivs = 10;
          break;
      }
    }

    auto driver = plot_cmd.get<std::string>("driver");
    if (driver == "gnuplot") {
      GnuplotPrinter gp{std::cout};
      std::visit(
          [&](auto&& intg) {
            intg.load(file);
            const S ndim = intg.ndim();
            if (nsamples == 0) {
              nsamples = 100 * ipow(ndivs, 2) * ipow(ndim + 1, 2);
            }
            intg.print(gp);
            gp << "\n";
            GnuplotSample<num_traits> sample(intg.id, ndim, ndivs, output);
            intg.integrate(sample, {.neval = nsamples,
                                    .niter = 1,
                                    .frozen = true,
                                    .verbosity = 0});
            sample.print(std::cout);
            std::cerr << std::format("# driver: \"{}\", ndim : {}, nsamples : {}, ndivs : {}\n",
                                     driver, ndim, nsamples, ndivs);
          },
          vint);
    } else {
      throw std::runtime_error("Unsupported driver");
    }

  }  // plot

  return 0;
}
