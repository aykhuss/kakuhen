#include "kakuhen/integrator/result.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "catch2/catch_approx.hpp"

TEST_CASE("accumulate", "[result]") {
  using kakuhen::integrator::Result;

  Result<double, uint32_t> result;
  using IntAcc = decltype(result)::int_acc_type;

  IntAcc intacc;
  for (int i = 1; i <= 10; ++i) {
    intacc.accumulate(static_cast<double>(i));
  }
  REQUIRE(intacc.count() == 10);
  REQUIRE(intacc.value() == Catch::Approx(11. / 2.));
  REQUIRE(intacc.variance() == Catch::Approx((55. / 6.) / 10.));  // variance of mean
  REQUIRE(intacc.error() == Catch::Approx(0.9574271077563381));

  result.accumulate(intacc);
  REQUIRE(result.size() == 1);
  REQUIRE(result.count() == 10);
  REQUIRE(result.value() == Catch::Approx(11. / 2.));
  REQUIRE(result.variance() == Catch::Approx((55. / 6.) / 10.));
  REQUIRE(result.error() == Catch::Approx(0.9574271077563381));

  intacc.reset();
  for (int i = -5; i <= 5; ++i) {
    intacc.accumulate(0.5 + i * 0.1);
  }
  REQUIRE(intacc.count() == 11);
  REQUIRE(intacc.value() == Catch::Approx(0.5));
  REQUIRE(intacc.variance() == Catch::Approx(0.01));
  REQUIRE(intacc.error() == Catch::Approx(0.1));

  result.accumulate(intacc);
  REQUIRE(result.size() == 2);
  REQUIRE(result.count() == 21);
  REQUIRE(result.value() == Catch::Approx(0.5539568345323741));
  REQUIRE(result.variance() == Catch::Approx(0.00989208633093526));
  REQUIRE(result.error() == Catch::Approx(0.09945896807696758));

  intacc.reset();
  for (int i = -3; i <= 3; ++i) {
    intacc.accumulate(0.6 + i * 0.05);
  }
  REQUIRE(intacc.count() == 7);
  REQUIRE(intacc.value() == Catch::Approx(0.6));
  REQUIRE(intacc.variance() == Catch::Approx(0.00166666666666667));
  REQUIRE(intacc.error() == Catch::Approx(0.04082482904638635));

  Result<double, uint32_t> result2;
  result2.accumulate(intacc);
  REQUIRE(result2.size() == 1);
  REQUIRE(result2.count() == 7);
  REQUIRE(result2.value() == Catch::Approx(0.6));
  REQUIRE(result2.variance() == Catch::Approx(0.00166666666666667));
  REQUIRE(result2.error() == Catch::Approx(0.04082482904638635));

  result.accumulate(result2);
  REQUIRE(result.size() == 3);
  REQUIRE(result.count() == 28);
  REQUIRE(result.value() == Catch::Approx(0.59336099585062241));
  REQUIRE(result.variance() == Catch::Approx(0.00142634854771785));
  REQUIRE(result.error() == Catch::Approx(0.03776702990331442));

}
