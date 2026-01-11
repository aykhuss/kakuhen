#include "kakuhen/util/hash.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>

TEST_CASE("Hash for different Plain Data", "[hash]") {
  using kakuhen::util::Hash;
  using kakuhen::util::HashValue_t;

  HashValue_t hash_int = Hash().add<int>(42).value();
  REQUIRE(hash_int == HashValue_t(10203658981158674303ull));

  HashValue_t hash_dble = Hash().add<double>(1e23).value();
  REQUIRE(hash_dble == HashValue_t(3556694915024222193ull));

  HashValue_t hash_comp = Hash().add<float>(3.3f).add<uint64_t>(99).add<bool>(false).value();
  REQUIRE(hash_comp == HashValue_t(7636397818777378217ull));
}

TEST_CASE("Hash for arrays and vectors", "[hash]") {
  using kakuhen::util::Hash;
  using kakuhen::util::HashValue_t;

  std::vector<double> vec{1.0, 2.0, 3.0, 4.0};
  HashValue_t hash_arr = Hash().add(vec.data(), vec.size()).value();
  REQUIRE(hash_arr == HashValue_t(10642777788671099552ull));

  HashValue_t hash_vec = Hash().add(std::vector<int>{1, 2, 3, 4}).value();
  REQUIRE(hash_vec == HashValue_t(9566659391000707361ull));
}

TEST_CASE("Hash to HEX representation", "[hash]") {
  using kakuhen::util::Hash;
  using kakuhen::util::HashValue_t;

  Hash h = Hash().add<double>(2.3).add<uint64_t>(666).add<bool>(true).add(std::vector<int>{4, 2});

  HashValue_t hash_comp = h.value();
  std::string hex = h.encode_hex();
  REQUIRE(hash_comp == HashValue_t(1325280494992402884ull));
  REQUIRE(hex == "12645796abc6a9c4");
}
