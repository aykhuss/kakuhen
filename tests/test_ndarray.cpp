#include "kakuhen/ndarray/ndarray.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <cstdint>
#include <sstream>

#include "catch2/catch_approx.hpp"

using Catch::Matchers::Message;
using Catch::Matchers::RangeEquals;

TEST_CASE("NDArray fill and access", "[ndarray]") {
  kakuhen::ndarray::NDArray<int> scalar_arr;
  REQUIRE(scalar_arr.size() == 1);
  scalar_arr.fill(42);
  /// scalars need to be accessed with [0]
  REQUIRE(scalar_arr[0] == 42);
  scalar_arr[0] = 23;
  REQUIRE(scalar_arr[0] == 23);

  kakuhen::ndarray::NDArray<int> arr({2, 3, 4});
  using size_type = decltype(arr)::size_type;
  std::vector<size_type> shape = arr.shape();
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
  std::vector<size_type> shape = arr.shape();
  arr.fill(-1);

  // std::cout << "---" << std::endl;
  // for (auto i0 = 0; i0 < shape[0]; ++i0) {
  //   for (auto i1 = 0; i1 < shape[1]; ++i1) {
  //     std::cout << "[" << i0 << "," << i1 << ",:" << "] = ";
  //     for (auto i2 = 0; i2 < shape[2]; ++i2) {
  //       std::cout << arr(i0, i1, i2) << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  auto view = arr.slice({{2, _}, {_, 4}, {_, _, 2}});
  using view_size_type = decltype(view)::size_type;
  STATIC_REQUIRE(std::is_same_v<view_size_type, size_type>);
  REQUIRE(view.ndim() == 3);
  REQUIRE_THAT(view.shape(), RangeEquals({3, 4, 16}));
  std::vector<size_type> vshape = view.shape();
  for (auto i0 = 0; i0 < vshape[0]; ++i0) {
    for (auto i1 = 0; i1 < vshape[1]; ++i1) {
      for (auto i2 = 0; i2 < vshape[2]; ++i2) {
        view(i0, i1, i2) = i0 + i1 * 10 + i2 * 100 + 10000;
      }
    }
  }

  // std::cout << "---" << std::endl;
  // for (auto i0 = 0; i0 < shape[0]; ++i0) {
  //   for (auto i1 = 0; i1 < shape[1]; ++i1) {
  //     std::cout << "[" << i0 << "," << i1 << ",:" << "] = ";
  //     for (auto i2 = 0; i2 < shape[2]; ++i2) {
  //       std::cout << arr(i0, i1, i2) << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }
}

TEST_CASE("NDView slice and access", "[ndarray]") {
  using kakuhen::ndarray::_;

  kakuhen::ndarray::NDArray<int> arr({4, 5, 6, 7, 8});
  using size_type = decltype(arr)::size_type;
  std::vector<size_type> shape = arr.shape();
  arr.fill(1);
  auto view = arr.slice({{1, 3}, {}, {_, 3}, {2, _}, {1, 7, 2}});
  std::vector<size_type> vshape = view.shape();

  REQUIRE(view.ndim() == 5);
  REQUIRE_THAT(vshape, RangeEquals({2, 5, 3, 5, 3}));

  for (auto i0 = 0; i0 < vshape[0]; ++i0) {
    for (auto i1 = 0; i1 < vshape[1]; ++i1) {
      for (auto i2 = 0; i2 < vshape[2]; ++i2) {
        for (auto i3 = 0; i3 < vshape[3]; ++i3) {
          for (auto i4 = 0; i4 < vshape[4]; ++i4) {
            REQUIRE(view(i0, i1, i2, i3, i4) == 1);
            view(i0, i1, i2, i3, i4) = i0 + i1 * 10 + i2 * 100 + i3 * 1000 + i4 * 10000;
          }
        }
      }
    }
  }

  REQUIRE(arr(1, 0, 0, 2, 1) == 0);
  REQUIRE(arr(2, 2, 1, 5, 3) == 1 + 2 * 10 + 1 * 100 + 3 * 1000 + 1 * 10000);

  arr = kakuhen::ndarray::NDArray<int, size_type>{3, 4, 5, 6, 7, 8, 9, 10, 13};
  shape = arr.shape();
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

  // for (auto idx = 0; idx < arr.size(); ++idx) {
  //   std::cout << arr[idx] << " ";
  // }
  // std::cout << std::endl;

  // for (auto i0 = 0; i0 < shape[0]; ++i0) {
  //   for (auto i1 = 0; i1 < shape[1]; ++i1) {
  //     for (auto i2 = 0; i2 < shape[2]; ++i2) {
  //       for (auto i3 = 0; i3 < shape[3]; ++i3) {
  //         for (auto i4 = 0; i4 < shape[4]; ++i4) {
  //           for (auto i5 = 0; i5 < shape[5]; ++i5) {
  //             for (auto i6 = 0; i6 < shape[6]; ++i6) {
  //               for (auto i7 = 0; i7 < shape[7]; ++i7) {
  //                 for (auto i8 = 0; i8 < shape[8]; ++i8) {
  //                   if (i0 == 1 && i1 == 2 && i2 == 2 && i3 == 3 && i4 == 3
  //                   &&
  //                       i5 == 3 && i6 == 3 && i7 == 4 && i8 == 4) {
  //                     REQUIRE(arr(i0, i1, i2, i3, i4, i5, i6, i7, i8) ==
  //                     1337);
  //                     // if (arr(i0, i1, i2, i3, i4, i5, i6, i7, i8) != 1337)
  //                     {
  //                     //   std::cout << "arr(" << i0 << ", " << i1 << ", " <<
  //                     i2
  //                     //             << ", " << i3 << ", " << i4 << ", " <<
  //                     i5
  //                     //             << ", " << i6 << ", " << i7 << ", " <<
  //                     i8
  //                     //             << ") != 1337" << std::endl;
  //                     //   std::cin.ignore();
  //                     // }
  //                   } else {
  //                     REQUIRE(arr(i0, i1, i2, i3, i4, i5, i6, i7, i8) == 1);
  //                     // if (arr(i0, i1, i2, i3, i4, i5, i6, i7, i8) != 1) {
  //                     //   std::cout << "arr(" << i0 << ", " << i1 << ", " <<
  //                     i2
  //                     //             << ", " << i3 << ", " << i4 << ", " <<
  //                     i5
  //                     //             << ", " << i6 << ", " << i7 << ", " <<
  //                     i8
  //                     //             << ") != 1" << std::endl;
  //                     //   std::cin.ignore();
  //                     // }
  //                   }
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  REQUIRE(arr(1, 2, 2, 3, 3, 3, 3, 4, 4) == 1337);
}

TEST_CASE("NDView slice of slice", "[ndarray]") {
  using kakuhen::ndarray::_;

  kakuhen::ndarray::NDArray<double, uint64_t> arr({5, 7, 32});
  using size_type = decltype(arr)::size_type;
  STATIC_REQUIRE(std::is_same_v<size_type, uint64_t>);
  std::vector<size_type> shape = arr.shape();
  arr.fill(77.7);

  auto view = arr.slice({{2, _}, {_, 4}, {_, _, 2}});
  using view_size_type = decltype(view)::size_type;
  STATIC_REQUIRE(std::is_same_v<view_size_type, size_type>);
  REQUIRE(view.ndim() == 3);
  REQUIRE_THAT(view.shape(), RangeEquals({3, 4, 16}));
  std::vector<size_type> vshape = view.shape();
  for (auto i0 = 0; i0 < vshape[0]; ++i0) {
    for (auto i1 = 0; i1 < vshape[1]; ++i1) {
      for (auto i2 = 0; i2 < vshape[2]; ++i2) {
        view(i0, i1, i2) = i0 + i1 * 1e-1 + i2 * 1e-2;
      }
    }
  }

  auto vview = view.slice({{_, 2}, {_, _, 2}, {12, _, 2}});
  using vview_size_type = decltype(vview)::size_type;
  STATIC_REQUIRE(std::is_same_v<vview_size_type, size_type>);
  REQUIRE(vview.ndim() == 3);
  REQUIRE_THAT(vview.shape(), RangeEquals({2, 2, 2}));
  std::vector<size_type> vvshape = vview.shape();
  for (auto i0 = 0; i0 < vvshape[0]; ++i0) {
    for (auto i1 = 0; i1 < vvshape[1]; ++i1) {
      for (auto i2 = 0; i2 < vvshape[2]; ++i2) {
        vview(i0, i1, i2) = -(i0 + i1 * 1e-1 + i2 * 1e-2);
      }
    }
  }

  // std::cout << "---" << std::endl;
  // for (auto i0 = 0; i0 < shape[0]; ++i0) {
  //   for (auto i1 = 0; i1 < shape[1]; ++i1) {
  //     std::cout << "[" << i0 << "," << i1 << ",:" << "] = ";
  //     for (auto i2 = 0; i2 < shape[2]; ++i2) {
  //       std::cout << arr(i0, i1, i2) << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  REQUIRE(arr(3, 2, 28) == Catch::Approx(-1.11));
}

TEST_CASE("NDView reshape & diagonal", "[ndarray]") {
  using kakuhen::ndarray::_;

  kakuhen::ndarray::NDArray<int> arr({3, 3, 2});
  using size_type = decltype(arr)::size_type;
  std::vector<size_type> shape = arr.shape();
  arr.fill(1);

  auto view = arr.slice({{}, {}, {}});
  REQUIRE(view.ndim() == 3);
  REQUIRE_THAT(view.shape(), RangeEquals({3, 3, 2}));
  std::vector<size_type> vshape = view.shape();
  for (auto i0 = 0; i0 < vshape[0]; ++i0) {
    for (auto i1 = 0; i1 < vshape[1]; ++i1) {
      for (auto i2 = 0; i2 < vshape[2]; ++i2) {
        view(i0, i1, i2) = i0 * 100 + i1 * 10 + i2;
      }
    }
  }

  auto view2d = view.reshape({3,6});
  using view2d_size_type = decltype(view2d)::size_type;
  STATIC_REQUIRE(std::is_same_v<view2d_size_type, size_type>);
  REQUIRE(view2d.ndim() == 2);
  REQUIRE_THAT(view2d.shape(), RangeEquals({3, 6}));
  std::vector<size_type> shape2d = view2d.shape();
  for (auto i0 = 0; i0 < shape2d[0]; ++i0) {
    for (auto i1 = 0; i1 < shape2d[1]; ++i1) {
      REQUIRE(view2d(i0, i1) == i0*100+(i1/2)*10+(i1%2));
    }
  }

  auto viewd = view.diagonal(0,1);
  using viewd_size_type = decltype(viewd)::size_type;
  STATIC_REQUIRE(std::is_same_v<viewd_size_type, size_type>);
  REQUIRE(viewd.ndim() == 2);
  REQUIRE_THAT(viewd.shape(), RangeEquals({3, 2}));
  std::vector<size_type> shaped = viewd.shape();
  for (auto i0 = 0; i0 < shaped[0]; ++i0) {
    for (auto i1 = 0; i1 < shaped[1]; ++i1) {
      REQUIRE(viewd(i0, i1) == view(i0, i0, i1));
    }
  }

}

TEST_CASE("NDArray serialization", "[ndarray]") {
  std::stringstream ss;

  kakuhen::ndarray::NDArray<float> arr({2, 3});
  arr.fill(23.42);
  arr.serialize(ss);

  kakuhen::ndarray::NDArray<float> arr_read;
  arr_read.deserialize(ss);

  for (auto i = 0; i < arr.size(); ++i) {
    REQUIRE(arr[i] == arr_read[i]);
  }

  /// test type mismatch
  ss.str("");
  arr.serialize(ss, true);  // add type info
  kakuhen::ndarray::NDArray<double> arr_mismatch;
  REQUIRE_THROWS_MATCHES(arr_mismatch.deserialize(ss, true), std::runtime_error,
                         Message("type or size mismatch for typename T"));
  ss.str("");
  arr.serialize(ss, true);  // add type info
  kakuhen::ndarray::NDArray<float, int> arr_mismatch2;
  REQUIRE_THROWS_MATCHES(arr_mismatch2.deserialize(ss, true), std::runtime_error,
                         Message("type or size mismatch for typename S"));
}
