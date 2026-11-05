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
  test_atomic_saves.cpp
  test_axis.cpp
  test_basin.cpp
  test_basin_generator.cpp
  test_algorithm.cpp
  test_generator.cpp
  test_envelope_files.cpp
  test_envelope_data.cpp
  test_hash.cpp
  test_histogram.cpp
  test_histogram_registry.cpp
  test_histogram_writer.cpp
  test_histogram_view.cpp
  test_histogram_serialize.cpp
  test_integrator_validation.cpp
  test_math.cpp
  test_map_point.cpp
  test_ndarray.cpp
  test_ndview.cpp
  test_printer.cpp
  test_progress_callback.cpp
  test_result.cpp
  test_serialize.cpp
  test_small_vector.cpp
  test_user_data.cpp
  test_vegas.cpp
  test_vegas_generator.cpp
  test_coverage_misc.cpp
)

add_executable(unit_tests ${TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE kakuhen kakuhen_warnings Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(unit_tests)

# replaces the global allocation functions, so it can't share a process with unit_tests
add_executable(allocation_tests test_allocations.cpp)
target_link_libraries(allocation_tests PRIVATE kakuhen kakuhen_warnings Catch2::Catch2WithMain)
catch_discover_tests(allocation_tests TEST_PREFIX "allocations: ")

# inside `kakuhen/build/tests` run `make && ctest`
