#include "kakuhen/kakuhen.h"
#include "kakuhen/util/printer.h"
#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>
#include <sstream>

using namespace kakuhen::integrator;

int main(int argc, char* argv[]) {

  kakuhen::util::printer::JSONPrinter jp{std::cout, 2};

  auto integrator = Basin(2,2,4);
  //auto integrator = Vegas(2, 6);
  integrator.print(jp);
  std::cout << std::endl;

  argparse::ArgumentParser program("kakuhen");

  //> kakuhen dump subparser
  argparse::ArgumentParser dump_cmd("dump");
  dump_cmd.add_description("dump the information of a kakuhen state file");
  dump_cmd.add_argument("file").help("kakuhen state file").nargs(1);  // exactly one file
  // -i INDENT, --indent INDENT

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
    auto vegas = Vegas<num_traits_t<float>>(file);
    vegas.print_grid();
  }

  return 0;
}
