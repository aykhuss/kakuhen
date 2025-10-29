#include "kakuhen/kakuhen.h"
#include "kakuhen/util/printer.h"
#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>
#include <variant>

using namespace kakuhen::integrator;

using Plain_t = Plain<>;
using Vegas_t = Vegas<>;
using Basin_t = Basin<>;
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
  using namespace kakuhen::util::type;

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
    kakuhen::util::printer::JSONPrinter jp{std::cout, indent};
    auto vint = make_integrator(Plain_t::parse_header(file));
    std::visit(
        [&](auto&& integrator) {
          integrator.load(file);
          integrator.print(jp);
          jp << "\n";
        },
        vint);
  }

  return 0;
}
