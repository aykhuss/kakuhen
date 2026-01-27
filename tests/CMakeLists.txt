include(FetchContent)

#> Catch2 for our test framework
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.9.0
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(catch2)

set(TEST_SOURCES
  test_accumulator.cpp
  test_axis.cpp
  test_basin.cpp
  test_hash.cpp
  test_histogram.cpp
  test_histogram_registry.cpp
  test_histogram_view.cpp
  test_histogram_serialize.cpp
  test_math.cpp
  test_ndarray.cpp
  test_printer.cpp
  test_result.cpp
  test_serialize.cpp
  test_user_data.cpp
  test_vegas.cpp
  test_coverage_misc.cpp
)

add_executable(unit_tests ${TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE kakuhen kakuhen_warnings Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(unit_tests)

# inside `kakuhen/build/tests` run `make && ctest`
