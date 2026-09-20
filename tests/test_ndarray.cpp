#include "kakuhen/ndarray/ndarray.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "catch2/catch_approx.hpp"

using Catch::Matchers::Message;
using Catch::Matchers::RangeEquals;

TEST_CASE("NDArray fill and access", "[ndarray]") {
  kakuhen::ndarray::NDArray<int> scalar_arr;
  REQUIRE(scalar_arr.size() == 0);

  kakuhen::ndarray::NDArray<int> arr({2, 3, 4});
  auto shape = arr.shape();
  REQUIRE(shape.size() == 3);
  REQUIRE_THAT(shape, RangeEquals({2, 3, 4}));
  arr.fill(42);
  REQUIRE(arr(0, 2, 2) == 42);
  arr(1, 2, 0) = 23;
  REQUIRE(arr(1, 2, 0) == 23);
}

TEST_CASE("NDView consistency", "[ndarray]") {
  using kakuhen::ndarray::_;

  kakuhen::ndarray::NDArray<int, uint16_t> arr({5, 7, 32});
  using size_type = decltype(arr)::size_type;
  STATIC_REQUIRE(std::is_same_v<size_type, uint16_t>);
  arr.fill(-1);

  auto view = arr.slice({{2, _}, {_, 4}, {_, _, 2}});
  using view_size_type = decltype(view)::size_type;
  STATIC_REQUIRE(std::is_same_v<view_size_type, size_type>);
  REQUIRE(view.ndim() == 3);
  REQUIRE_THAT(view.shape(), RangeEquals({3, 4, 16}));
  auto vshape = view.shape();
  for (size_t i0 = 0; i0 < vshape[0]; ++i0) {
    for (size_t i1 = 0; i1 < vshape[1]; ++i1) {
      for (size_t i2 = 0; i2 < vshape[2]; ++i2) {
        view(i0, i1, i2) = static_cast<int>(i0 + i1 * 10 + i2 * 100 + 10000);
      }
    }
  }
}

TEST_CASE("NDView slice and access", "[ndarray]") {
  using kakuhen::ndarray::_;

  kakuhen::ndarray::NDArray<int> arr({4, 5, 6, 7, 8});
  using size_type = decltype(arr)::size_type;
  arr.fill(1);
  auto view = arr.slice({{1, 3}, {}, {_, 3}, {2, _}, {1, 7, 2}});
  auto vshape = view.shape();

  REQUIRE(view.ndim() == 5);
  REQUIRE_THAT(vshape, RangeEquals({2, 5, 3, 5, 3}));

  for (size_t i0 = 0; i0 < vshape[0]; ++i0) {
    for (size_t i1 = 0; i1 < vshape[1]; ++i1) {
      for (size_t i2 = 0; i2 < vshape[2]; ++i2) {
        for (size_t i3 = 0; i3 < vshape[3]; ++i3) {
          for (size_t i4 = 0; i4 < vshape[4]; ++i4) {
            REQUIRE(view(i0, i1, i2, i3, i4) == 1);
            view(i0, i1, i2, i3, i4) =
                static_cast<int>(i0 + i1 * 10 + i2 * 100 + i3 * 1000 + i4 * 10000);
          }
        }
      }
    }
  }

  REQUIRE(arr(1, 0, 0, 2, 1) == 0);
  REQUIRE(arr(2, 2, 1, 5, 3) == 1 + 2 * 10 + 1 * 100 + 3 * 1000 + 1 * 10000);

  arr = kakuhen::ndarray::NDArray<int, size_type>{3, 4, 5, 6, 7, 8, 9, 10, 13};
  arr.fill(1);
  REQUIRE(arr(0, 1, 2, 3, 4, 5, 6, 7, 8) == 1);

  view = arr.slice({{_, _, _},     // [0]
                    {0, _, 2},     // [1]
                    {_, _, 2},     // [2]
                    {1, _, 2},     // [3]
                    {1, _, 2},     // [4]
                    {_, _, 3},     // [5]
                    {0, _, 3},     // [6]
                    {1, _, 3},     // [7]
                    {1, 11, 3}});  // [8]
  vshape = view.shape();
  REQUIRE_THAT(vshape, RangeEquals({3, 2, 3, 3, 3, 3, 3, 3, 4}));
  view(1, 1, 1, 1, 1, 1, 1, 1, 1) = 1337;

  REQUIRE(arr(1, 2, 2, 3, 3, 3, 3, 4, 4) == 1337);
}

TEST_CASE("NDView slice of slice", "[ndarray]") {
  using kakuhen::ndarray::_;

  kakuhen::ndarray::NDArray<double, uint64_t> arr({5, 7, 32});
  using size_type = decltype(arr)::size_type;
  STATIC_REQUIRE(std::is_same_v<size_type, uint64_t>);
  arr.fill(77.7);

  auto view = arr.slice({{2, _}, {_, 4}, {_, _, 2}});
  using view_size_type = decltype(view)::size_type;
  STATIC_REQUIRE(std::is_same_v<view_size_type, size_type>);
  REQUIRE(view.ndim() == 3);
  REQUIRE_THAT(view.shape(), RangeEquals({3, 4, 16}));
  auto vshape = view.shape();
  for (size_t i0 = 0; i0 < vshape[0]; ++i0) {
    for (size_t i1 = 0; i1 < vshape[1]; ++i1) {
      for (size_t i2 = 0; i2 < vshape[2]; ++i2) {
        view(i0, i1, i2) = static_cast<double>(i0) + i1 * 1e-1 + i2 * 1e-2;
      }
    }
  }

  auto vview = view.slice({{_, 2}, {_, _, 2}, {12, _, 2}});
  using vview_size_type = decltype(vview)::size_type;
  STATIC_REQUIRE(std::is_same_v<vview_size_type, size_type>);
  REQUIRE(vview.ndim() == 3);
  REQUIRE_THAT(vview.shape(), RangeEquals({2, 2, 2}));
  auto vvshape = vview.shape();
  for (size_t i0 = 0; i0 < vvshape[0]; ++i0) {
    for (size_t i1 = 0; i1 < vvshape[1]; ++i1) {
      for (size_t i2 = 0; i2 < vvshape[2]; ++i2) {
        vview(i0, i1, i2) = -(static_cast<double>(i0) + i1 * 1e-1 + i2 * 1e-2);
      }
    }
  }

  REQUIRE(arr(3, 2, 28) == Catch::Approx(-1.11));
}

TEST_CASE("NDView reshape & diagonal", "[ndarray]") {
  using kakuhen::ndarray::_;

  kakuhen::ndarray::NDArray<int> arr({3, 3, 2});
  using size_type = decltype(arr)::size_type;
  arr.fill(1);

  auto view = arr.slice({{}, {}, {}});
  REQUIRE(view.ndim() == 3);
  REQUIRE_THAT(view.shape(), RangeEquals({3, 3, 2}));
  auto vshape = view.shape();
  for (size_t i0 = 0; i0 < vshape[0]; ++i0) {
    for (size_t i1 = 0; i1 < vshape[1]; ++i1) {
      for (size_t i2 = 0; i2 < vshape[2]; ++i2) {
        view(i0, i1, i2) = static_cast<int>(i0 * 100 + i1 * 10 + i2);
      }
    }
  }

  auto view2d = view.reshape({3, 6});
  using view2d_size_type = decltype(view2d)::size_type;
  STATIC_REQUIRE(std::is_same_v<view2d_size_type, size_type>);
  REQUIRE(view2d.ndim() == 2);
  REQUIRE_THAT(view2d.shape(), RangeEquals({3, 6}));
  auto shape2d = view2d.shape();
  for (size_t i0 = 0; i0 < shape2d[0]; ++i0) {
    for (size_t i1 = 0; i1 < shape2d[1]; ++i1) {
      REQUIRE(view2d(i0, i1) == static_cast<int>(i0 * 100 + (i1 / 2) * 10 + (i1 % 2)));
    }
  }

  auto viewd = view.diagonal(0, 1);
  using viewd_size_type = decltype(viewd)::size_type;
  STATIC_REQUIRE(std::is_same_v<viewd_size_type, size_type>);
  REQUIRE(viewd.ndim() == 2);
  REQUIRE_THAT(viewd.shape(), RangeEquals({3, 2}));
  auto shaped = viewd.shape();
  for (size_t i0 = 0; i0 < shaped[0]; ++i0) {
    for (size_t i1 = 0; i1 < shaped[1]; ++i1) {
      REQUIRE(viewd(i0, i1) == view(i0, i0, i1));
    }
  }
}

TEST_CASE("NDArray serialization", "[ndarray]") {
  std::stringstream ss;

  kakuhen::ndarray::NDArray<float> arr({2, 3});
  arr.fill(23.42f);
  arr.serialize(ss);

  kakuhen::ndarray::NDArray<float> arr_read;
  arr_read.deserialize(ss);

  for (size_t i = 0; i < static_cast<size_t>(arr.size()); ++i) {
    REQUIRE(arr[static_cast<uint32_t>(i)] == arr_read[static_cast<uint32_t>(i)]);
  }

  /// test type mismatch
  ss.str("");
  ss.clear();
  arr.serialize(ss, true);  // add type info
  kakuhen::ndarray::NDArray<double> arr_mismatch;
  REQUIRE_THROWS_MATCHES(arr_mismatch.deserialize(ss, true), std::runtime_error,
                         Message("type or size mismatch for typename T"));
  ss.str("");
  ss.clear();
  arr.serialize(ss, true);  // add type info
  kakuhen::ndarray::NDArray<float, int> arr_mismatch2;
  REQUIRE_THROWS_MATCHES(arr_mismatch2.deserialize(ss, true), std::runtime_error,
                         Message("type or size mismatch for typename S"));
}

TEST_CASE("NDArray deserialization against an expected shape", "[ndarray]") {
  using S = uint32_t;
  std::stringstream ss;
  kakuhen::ndarray::NDArray<float> arr({2, 3});
  arr.fill(23.42f);
  arr.serialize(ss);
  const std::string bytes = ss.str();

  std::stringstream in(bytes);
  kakuhen::ndarray::NDArray<float> arr_read;
  arr_read.deserialize_expected_shape(in, std::array<S, 2>{2, 3});
  REQUIRE_THAT(arr_read, RangeEquals(arr));

  SECTION("C-style shape arrays") {
    const S shape[] = {2, 3};
    std::stringstream matching(bytes);
    arr_read.deserialize_expected_shape(matching, shape);
    REQUIRE_THAT(arr_read, RangeEquals(arr));

    const S wrong_shape[] = {3, 2};
    std::stringstream mismatching(bytes);
    REQUIRE_THROWS_MATCHES(arr_read.deserialize_expected_shape(mismatching, wrong_shape),
                           std::runtime_error,
                           Message("NDArray: shape mismatch during deserialization"));
  }

  for (const auto& shape : {std::vector<S>{2, 4}, std::vector<S>{2}, std::vector<S>{2, 3, 1}}) {
    std::stringstream mismatch(bytes);
    REQUIRE_THROWS_MATCHES(arr_read.deserialize_expected_shape(mismatch, shape), std::runtime_error,
                           Message("NDArray: shape mismatch during deserialization"));
  }

  // a header whose element count overflows is rejected before any allocation
  const S huge = std::numeric_limits<S>::max();
  std::stringstream overflow;
  kakuhen::util::serialize::serialize_one<S>(overflow, 2);
  kakuhen::util::serialize::serialize_one<S>(overflow, huge);
  kakuhen::util::serialize::serialize_one<S>(overflow, huge);
  REQUIRE_THROWS_MATCHES(arr_read.deserialize_expected_shape(overflow, std::array{huge, huge}),
                         std::runtime_error, Message("NDArray: size overflow"));
}

TEST_CASE("NDArray can deserialize using its own shape", "[ndarray]") {
  kakuhen::ndarray::NDArray<double> arr({2, 3});
  arr.fill(42.0);
  std::stringstream serialized;
  arr.serialize(serialized);
  arr.fill(7.0);
  const std::string bytes = serialized.str();

  SECTION("successful replacement") {
    arr.deserialize_expected_shape(serialized, arr.shape());
    REQUIRE_THAT(arr.shape(), RangeEquals({2u, 3u}));
    for (double value : arr)
      REQUIRE(value == 42.0);
  }
  SECTION("truncated payload preserves the original array") {
    std::stringstream truncated(bytes.substr(0, bytes.size() - 1));
    REQUIRE_THROWS_AS(arr.deserialize_expected_shape(truncated, arr.shape()), std::runtime_error);
    REQUIRE_THAT(arr.shape(), RangeEquals({2u, 3u}));
    for (double value : arr)
      REQUIRE(value == 7.0);
  }
  SECTION("unconstrained loading also preserves the original array on failure") {
    std::stringstream truncated(bytes.substr(0, bytes.size() - 1));
    REQUIRE_THROWS_AS(arr.deserialize(truncated), std::runtime_error);
    REQUIRE_THAT(arr.shape(), RangeEquals({2u, 3u}));
    for (double value : arr)
      REQUIRE(value == 7.0);
  }
}

TEST_CASE("NDArray allocation checks include metadata", "[ndarray]") {
  using S = uint64_t;
  using Array = kakuhen::ndarray::NDArray<uint64_t, S>;
  const S extent = std::numeric_limits<size_t>::max() / sizeof(uint64_t);
  SECTION("construction") {
    REQUIRE_THROWS_MATCHES(Array({extent}), std::runtime_error, Message("NDArray: size overflow"));
  }
  SECTION("deserialization") {
    std::stringstream in;
    kakuhen::util::serialize::serialize_one<S>(in, 1);
    kakuhen::util::serialize::serialize_one<S>(in, extent);
    kakuhen::util::serialize::serialize_one<S>(in, extent);
    Array arr({2});
    arr.fill(17);
    SECTION("expected shape") {
      REQUIRE_THROWS_MATCHES(arr.deserialize_expected_shape(in, std::array{extent}),
                             std::runtime_error, Message("NDArray: size overflow"));
    }
    SECTION("shape read from stream") {
      REQUIRE_THROWS_MATCHES(arr.deserialize(in), std::runtime_error,
                             Message("NDArray: size overflow"));
    }
    REQUIRE(arr.size() == 2);
    REQUIRE(arr[0] == 17);
    REQUIRE(arr[1] == 17);
  }
}

TEST_CASE("NDArray empty shapes allow large representable strides", "[ndarray]") {
  using S = uint64_t;
  using Array = kakuhen::ndarray::NDArray<double, S>;
  const S huge = std::numeric_limits<S>::max();
  const S shape[] = {0, huge};
  Array arr(std::span<const S>{shape});
  REQUIRE(arr.empty());
  REQUIRE_THAT(arr.shape(), RangeEquals(shape));
  REQUIRE_THAT(arr.strides(), RangeEquals(std::array<S, 2>{huge, 1}));

  std::stringstream serialized;
  arr.serialize(serialized);
  Array restored;
  restored.deserialize_expected_shape(serialized, shape);
  REQUIRE(restored.empty());
  REQUIRE_THAT(restored.shape(), RangeEquals(shape));
  REQUIRE_THAT(restored.strides(), RangeEquals(arr.strides()));

  // A zero extent must not conceal overflow in a trailing stride.
  REQUIRE_THROWS_MATCHES(Array({0, 2, huge}), std::runtime_error,
                         Message("NDArray: size overflow"));
}
