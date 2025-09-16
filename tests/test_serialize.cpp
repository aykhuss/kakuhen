#include "kakuhen/util/serialize.h"
#include <catch2/catch_test_macros.hpp>
#include <sstream>

using namespace kakuhen::util::serialize;

TEST_CASE("Write and read one POD value", "[serialize]") {
  std::stringstream ss;

  int orig_int = 42;
  serialize_one(ss, orig_int);
  int back_int = 0;
  deserialize_one(ss, back_int);
  REQUIRE(back_int == orig_int);

  double orig_dble = 42;
  serialize_one(ss, orig_dble);
  double back_dble = 0;
  deserialize_one(ss, back_dble);
  REQUIRE(back_dble == orig_dble);
}

TEST_CASE("Write and read array of POD values", "[serialize]") {
  std::stringstream ss;

  float arr[5] = {1, 2, 3, 4, 5};
  serialize_array(ss, arr, 5);

  float read_arr[5] = {0};
  deserialize_array(ss, read_arr, 5);

  for (auto i = 0; i < 5; ++i) {
    REQUIRE(read_arr[i] == arr[i]);
  }
}

TEST_CASE("deserialize_one throws on short stream", "[serialize]") {
  std::stringstream ss;

  int value = 12345;
  serialize_one(ss, value);

  //> Truncate to simulate incomplete stream
  std::string truncated = ss.str().substr(0, sizeof(int) - 1);
  std::stringstream truncated_ss(truncated);

  int read_value = 0;
  REQUIRE_THROWS_AS(deserialize_one(truncated_ss, read_value),
                    std::runtime_error);
}

TEST_CASE("Write and read container of POD values", "[serialize]") {
  std::stringstream ss;

  std::vector<double> arr{1, 2, 3, 4, 5};
  serialize_container(ss, arr);

  // need to ensure memory is allocated with the right size
  std::vector<double> read_arr(5);
  deserialize_container(ss, read_arr);

  for (auto i = 0; i < arr.size(); ++i) {
    REQUIRE(read_arr[i] == arr[i]);
  }
}
