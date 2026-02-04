#include "gnuplot.h"
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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <variant>

using namespace kakuhen;
using namespace integrator;
using namespace histogram;
using namespace util::printer;

/// default numeric traits
using num_traits = util::num_traits_t<>;

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

  //> kakuhen sample subparser
  argparse::ArgumentParser sample_cmd("sample");
  sample_cmd.add_description("sample points using a kakuhen state file");
  sample_cmd.add_argument("file").help("kakuhen state file").nargs(1);  // exactly one file
  program.add_subparser(sample_cmd);

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
        [&](auto&& integrator) {
          integrator.load(file);
          integrator.print(jp);
          jp << "\n";
        },
        vint);
  }  // dump

  if (program.is_subcommand_used("sample")) {
    auto file = sample_cmd.get<std::string>("file");
    auto header = parse_header(file);
    auto vint = make_integrator(header);
    GnuplotPrinter gp{std::cout};
    std::visit(
        [&](auto&& intg) {
          intg.load(file);
          intg.print(gp);
          gp << "\n";
          GnuplotSample<num_traits> sample(intg.ndim(), 51);
          intg.integrate(sample, {.neval = 50000000, .niter = 1, .adapt = false, .verbosity = 0});
          sample.print();
        },
        vint);
  }  // sample

  return 0;
}
