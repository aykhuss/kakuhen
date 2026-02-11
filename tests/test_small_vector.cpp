#include <catch2/catch_test_macros.hpp>
#include "kakuhen/util/small_vector.h"
#include <vector>

using namespace kakuhen::util;

TEST_CASE("SmallVector: DefaultConstructor", "[util][small_vector]") {
  SmallVector<int, 4> v;
  REQUIRE(v.size() == 0);
  REQUIRE(v.capacity() == 4);
  REQUIRE(v.empty());
}

TEST_CASE("SmallVector: PushBackInline", "[util][small_vector]") {
  SmallVector<int, 4> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.push_back(4);
  REQUIRE(v.size() == 4);
  REQUIRE(v.capacity() == 4);
  REQUIRE(v[0] == 1);
  REQUIRE(v[3] == 4);
}

TEST_CASE("SmallVector: PushBackGrow", "[util][small_vector]") {
  SmallVector<int, 2> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  REQUIRE(v.size() == 3);
  REQUIRE(v.capacity() > 2);
  REQUIRE(v[0] == 1);
  REQUIRE(v[1] == 2);
  REQUIRE(v[2] == 3);
}

TEST_CASE("SmallVector: CopyConstructor", "[util][small_vector]") {
  SmallVector<int, 2> v1;
  v1.push_back(1);
  v1.push_back(2);
  v1.push_back(3);

  SmallVector<int, 2> v2 = v1;
  REQUIRE(v2.size() == 3);
  REQUIRE(v2[0] == 1);
  REQUIRE(v2[1] == 2);
  REQUIRE(v2[2] == 3);
}

TEST_CASE("SmallVector: MoveConstructor", "[util][small_vector]") {
  SmallVector<int, 2> v1;
  v1.push_back(1);
  v1.push_back(2);
  v1.push_back(3);
  const int* old_data = v1.data();

  SmallVector<int, 2> v2 = std::move(v1);
  REQUIRE(v2.size() == 3);
  REQUIRE(v2.data() == old_data);
  REQUIRE(v1.size() == 0);
}

TEST_CASE("SmallVector: Clear", "[util][small_vector]") {
  SmallVector<int, 2> v;
  v.push_back(1);
  v.clear();
  REQUIRE(v.size() == 0);
  REQUIRE(v.empty());
}

TEST_CASE("SmallVector: Resize", "[util][small_vector]") {
    SmallVector<int, 4> v;
    v.resize(2, 42);
    REQUIRE(v.size() == 2);
    REQUIRE(v[0] == 42);
    REQUIRE(v[1] == 42);
    
    v.resize(5, 10);
    REQUIRE(v.size() == 5);
    REQUIRE(v[4] == 10);
}

TEST_CASE("SmallVector: InitializerList", "[util][small_vector]") {
    SmallVector<int, 4> v = {1, 2, 3};
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 1);
    REQUIRE(v[1] == 2);
    REQUIRE(v[2] == 3);
}

TEST_CASE("SmallVector: PopBack", "[util][small_vector]") {
    SmallVector<int, 4> v = {1, 2, 3};
    v.pop_back();
    REQUIRE(v.size() == 2);
    REQUIRE(v.back() == 2);
}

TEST_CASE("SmallVector: EmplaceBack", "[util][small_vector]") {
    struct Point {
        int x, y;
    };
    SmallVector<Point, 4> v;
    v.emplace_back(1, 2);
    REQUIRE(v.size() == 1);
    REQUIRE(v[0].x == 1);
    REQUIRE(v[0].y == 2);
}