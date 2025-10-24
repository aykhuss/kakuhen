cmake_minimum_required(VERSION 3.18)
project(kakuhen VERSION 0.0.1 LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#> generate `compile_commands.json`
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#> enable tests
option(BUILD_TESTING "Enable creation of kakuhen tests." OFF)
#> enable building CLI tool
option(KAKUHEN_BUILD_CLI "Build kakuhen CLI tool" ON)

#> default build type is: Release
if (NOT CMAKE_BUILD_TYPE)
  #> other options: Debug, Release, RelWithDebInfo, MinSizeRel
  #set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
#> would have thought that CMake adds this automatically...
add_compile_definitions($<IF:$<CONFIG:Debug>,DEBUG,NDEBUG>)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

add_library(kakuhen INTERFACE)
add_library(kakuhen::kakuhen ALIAS kakuhen)

# target_compile_features(kakuhen INTERFACE cxx_std_17)
target_compile_features(kakuhen INTERFACE cxx_std_20)

target_include_directories(kakuhen INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/kakuhenConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/kakuhenConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kakuhen
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# {fmt} to avoid std::format
add_subdirectory(external/fmt EXCLUDE_FROM_ALL)
target_link_libraries(kakuhen INTERFACE fmt::fmt)

install(
  TARGETS kakuhen fmt
  EXPORT kakuhenTargets)
install(
  EXPORT kakuhenTargets
  FILE kakuhenTargets.cmake
  NAMESPACE kakuhen::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kakuhen
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/kakuhenConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kakuhen
)
install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/kakuhen
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

add_subdirectory(examples)
if(KAKUHEN_BUILD_CLI)
  add_subdirectory(tools)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
