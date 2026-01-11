#include "kakuhen/util/printer.h"
#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <vector>

using namespace kakuhen::util::printer;

TEST_CASE("JSONPrinter basic types", "[printer]") {
  std::stringstream ss;
  JSONPrinter p(ss); // Compact mode (indent = 0)

  SECTION("Integers") {
    p.begin<Context::OBJECT>();
    p.print_one("key", 42);
    p.end<Context::OBJECT>();
    REQUIRE(ss.str() == "{\"key\":42}");
  }

  SECTION("Floats") {
    p.begin<Context::OBJECT>();
    p.print_one("pi", 3.14);
    p.end<Context::OBJECT>();
    REQUIRE(ss.str() == "{\"pi\":3.14}");
  }

  SECTION("Strings") {
    p.begin<Context::OBJECT>();
    p.print_one("name", "kakuhen");
    p.end<Context::OBJECT>();
    REQUIRE(ss.str() == "{\"name\":\"kakuhen\"}");
  }
}

TEST_CASE("JSONPrinter containers", "[printer]") {
  std::stringstream ss;
  JSONPrinter p(ss);

  SECTION("Vector") {
    std::vector<int> vec = {1, 2, 3};
    p.begin<Context::OBJECT>();
    p.print_array("numbers", vec);
    p.end<Context::OBJECT>();
    REQUIRE(ss.str() == "{\"numbers\":[1,2,3]}");
  }

  SECTION("C-array") {
    int arr[] = {10, 20};
    p.begin<Context::OBJECT>();
    p.print_array("arr", arr, 2);
    p.end<Context::OBJECT>();
    REQUIRE(ss.str() == "{\"arr\":[10,20]}");
  }
}

TEST_CASE("JSONPrinter nested structure", "[printer]") {
  std::stringstream ss;
  JSONPrinter p(ss);

  // Expected: {"obj":{"val":1},"arr":[1,2]}
  p.begin<Context::OBJECT>();

  p.begin<Context::OBJECT>("obj");
  p.print_one("val", 1);
  p.end<Context::OBJECT>();

  p.begin<Context::ARRAY>("arr");
  p.print_one({}, 1);
  p.print_one({}, 2);
  p.end<Context::ARRAY>();

  p.end<Context::OBJECT>();

  REQUIRE(ss.str() == "{\"obj\":{\"val\":1},\"arr\":[1,2]}");
}

TEST_CASE("JSONPrinter indentation", "[printer]") {
  std::stringstream ss;
  JSONPrinter p(ss, 2); // Indent = 2 spaces

  p.begin<Context::OBJECT>();
  p.print_one("a", 1);
  p.end<Context::OBJECT>(true); // Pass true to force newline before closing brace

  std::string expected =
      "{\n"
      "  \"a\": 1\n"
      "}";

  REQUIRE(ss.str() == expected);
}

TEST_CASE("JSONPrinter string escaping", "[printer]") {
  std::stringstream ss;
  JSONPrinter p(ss);

  p.begin<Context::OBJECT>();
  p.print_one("key", "line\nbreak \"quote\" \\backslash");
  p.end<Context::OBJECT>();
  REQUIRE(ss.str() == "{\"key\":\"line\\nbreak \\\"quote\\\" \\\\backslash\"}");
}
