#include "kakuhen/util/user_data.h"
#include <catch2/catch_test_macros.hpp>
#include <sstream>

using namespace kakuhen::util;
using namespace kakuhen::util::serialize;

TEST_CASE("read/write user data stream", "[user_data]") {
  std::stringstream ss;

  struct UserData {
    int id;
    double value;
  };

  UserData ud_in = {42, 3.14};
  write_user_data_stream(ss, ud_in);

  UserData ud_out{};
  read_user_data_stream(ss, ud_out);

  REQUIRE(ud_out.id == ud_in.id);
  REQUIRE(ud_out.value == ud_in.value);
}
