#include "kakuhen/histogram/histogram_printer.h"
#include "kakuhen/histogram/histogram_registry.h"
#include "kakuhen/histogram/axis.h"
#include <catch2/catch_test_macros.hpp>
#include <sstream>

using namespace kakuhen::histogram;

TEST_CASE("NNLOJETPrinter output", "[printer]") {
  HistogramRegistry<> registry;
  UniformAxis<> x_ax(2, 0.0, 2.0); // 4 bins: U, [0,1), [1,2), O
  auto id = registry.book("test_hist", 1, x_ax);

  auto buffer = registry.create_buffer();
  registry.fill(buffer, id, 10.0, 0.5); // [0,1)
  registry.fill(buffer, id, 20.0, 1.5); // [1,2)
  registry.flush(buffer);

  std::stringstream ss;
  NNLOJETPrinter printer(ss);
  registry.print(printer);

  std::string output = ss.str();

  // Check headers
  REQUIRE(output.find("#name: test_hist") != std::string::npos);
  REQUIRE(output.find("#labels:") != std::string::npos);

  // Check bins (4 lines expected: UF, 2 regular, OF)
  // UF: (-inf, 0.0) -> mean 0
  // Bin 1: [0, 1) -> mean 10.0
  // Bin 2: [1, 2) -> mean 20.0
  // OF: [2, inf) -> mean 0

  REQUIRE(output.find("-inf") != std::string::npos);
  REQUIRE(output.find("inf") != std::string::npos);
  REQUIRE(output.find("1.0000000000e+01") != std::string::npos);
  REQUIRE(output.find("2.0000000000e+01") != std::string::npos);
}
