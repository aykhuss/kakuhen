#include "kakuhen/integrator/vegas.h"
#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace kakuhen::integrator;

int main(int argc, char* argv[]) {
  std::cout << "Hello, 世界!\n";

  argparse::ArgumentParser program("kakuhen");

  //> kakuhen dump subparser
  argparse::ArgumentParser dump_cmd("dump");
  dump_cmd.add_description("dump the information of a kakuhen state file");
  dump_cmd.add_argument("file").help("kakuhen state file to dump").nargs(1);  // exactly one file

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
