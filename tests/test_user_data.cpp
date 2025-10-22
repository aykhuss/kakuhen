#include "kakuhen/util/user_data.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <filesystem>
#include <sstream>

using Catch::Matchers::Message;

using namespace kakuhen::util;
using namespace kakuhen::util::serialize;

struct UserData {
  int id;
  double value;
};

struct TempFile {
  std::filesystem::path path;
  explicit TempFile(std::string_view file_name = "test.tmp")
      : path(std::filesystem::temp_directory_path() / std::string(file_name)) {}

  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  operator const std::filesystem::path&() const noexcept {
    return path;
  }
};

TEST_CASE("read/write user data stream", "[user_data]") {
  std::stringstream ss;

  UserData ud_in = {42, 3.14};
  write_user_data_stream(ss, ud_in);

  /// read in same data
  UserData ud_out{};
  read_user_data_stream(ss, ud_out);
  REQUIRE(ud_out.id == ud_in.id);
  REQUIRE(ud_out.value == ud_in.value);

  /// different headers should fail
  ss.clear();
  write_user_data_stream(ss, ud_in, "ASDF");
  REQUIRE_THROWS_MATCHES(read_user_data_stream(ss, ud_out), std::runtime_error,
                         Message("Incompatible user data headers"));
}

TEST_CASE("read/write user data file", "[user_data]") {
  TempFile tmp("test-user_data.bin");

  UserData ud1_in = {42, 3.14};
  UserData ud2_in = {23, 1.2};
  UserData ud3_in = {99, -666};

  write_user_data(tmp.path, ud1_in);
  write_user_data(tmp.path, ud2_in, "ASDF");
  write_user_data(tmp.path, ud3_in, "QWERTY");

  UserData ud1_out{};
  UserData ud2_out{};
  UserData ud3_out{};
  read_user_data(tmp.path, ud1_out);
  read_user_data(tmp.path, ud3_out, "QWERTY");
  read_user_data(tmp.path, ud2_out, "ASDF");

  REQUIRE(ud1_out.id == ud1_in.id);
  REQUIRE(ud1_out.value == ud1_in.value);
  REQUIRE(ud2_out.id == ud2_in.id);
  REQUIRE(ud2_out.value == ud2_in.value);
  REQUIRE(ud3_out.id == ud3_in.id);
  REQUIRE(ud3_out.value == ud3_in.value);
}
