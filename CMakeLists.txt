cmake_minimum_required(VERSION 3.18)
project(kakuhen VERSION 0.1.0 LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#> generate `compile_commands.json`
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#> enable tests
option(KAKUHEN_BUILD_TESTING "Enable creation of kakuhen tests." OFF)
#> enable building examples
option(KAKUHEN_BUILD_EXAMPLES "Build examples" OFF)
#> enable building CLI tool
option(KAKUHEN_BUILD_CLI "Build kakuhen CLI tool" OFF)
#> enable building documentation
option(KAKUHEN_BUILD_DOCS "Build project documentation" OFF)
#> enable code coverage
option(KAKUHEN_ENABLE_COVERAGE "Enable code coverage generation" OFF)

if(KAKUHEN_ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_library(kakuhen_coverage INTERFACE)
    target_compile_options(kakuhen_coverage INTERFACE --coverage)
    target_link_options(kakuhen_coverage INTERFACE --coverage)
  else()
    message(WARNING "Code coverage is only supported for GCC and Clang")
  endif()
endif() # End KAKUHEN_ENABLE_COVERAGE

#> default build type is: Release
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  #> other options: Debug, Release, RelWithDebInfo, MinSizeRel
  #set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
#> would have thought that CMake adds this automatically...
add_compile_definitions($<IF:$<CONFIG:Debug>,DEBUG,NDEBUG>)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

add_library(kakuhen_warnings INTERFACE)
set_project_warnings(kakuhen_warnings)

add_library(kakuhen INTERFACE)
add_library(kakuhen::kakuhen ALIAS kakuhen)

# target_compile_features(kakuhen INTERFACE cxx_std_17)
target_compile_features(kakuhen INTERFACE cxx_std_20)

target_include_directories(kakuhen INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(KAKUHEN_ENABLE_COVERAGE AND TARGET kakuhen_coverage)
  target_link_libraries(kakuhen INTERFACE $<BUILD_INTERFACE:kakuhen_coverage>)
endif()

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

# #> {fmt} to avoid std::format
# find_package(fmt QUIET)
# if(NOT fmt_FOUND)
#   message(STATUS "Using bundled fmt from external/fmt")
#   # set(FMT_INSTALL OFF CACHE BOOL "Disable fmt installation" FORCE)
#   set(FMT_INSTALL ON CACHE BOOL "Enable fmt installation" FORCE)
#   set(FMT_HEADER_ONLY ON CACHE BOOL "Use header-only fmt" FORCE)
#   add_subdirectory(external/fmt EXCLUDE_FROM_ALL)
# endif()
# target_link_libraries(kakuhen INTERFACE fmt::fmt)

install(
  TARGETS kakuhen
  EXPORT kakuhenTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
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

if(KAKUHEN_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(KAKUHEN_BUILD_CLI)
  add_subdirectory(tools)
endif()
set(BUILD_TESTING ${KAKUHEN_BUILD_TESTING} CACHE BOOL "Build tests" FORCE)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(KAKUHEN_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  find_package(Sphinx REQUIRED)

  message(STATUS "Doxygen found: ${DOXYGEN_EXECUTABLE}")
  message(STATUS "Sphinx found: ${SPHINX_EXECUTABLE}")

  # Configure Doxyfile from Doxyfile.in
  configure_file(
          ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
          ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile # Output Doxyfile to build directory
          @ONLY
      )

  # Configure Sphinx conf.py from conf.py.in
  configure_file(
          ${CMAKE_CURRENT_SOURCE_DIR}/docs/sphinx/conf.py.in
          ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx/conf.py
          @ONLY
      )

  # Doxygen target
  add_custom_target(generate_doxygen_xml
          COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile # Use the generated Doxyfile
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          COMMENT "Generating Doxygen XML documentation"
          VERBATIM
      )

  # Sphinx target
  add_custom_target(build_sphinx_html
            COMMAND Sphinx::Build -b html
                    -c ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx # Location of conf.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/docs/sphinx # Source directory
                    ${CMAKE_CURRENT_SOURCE_DIR}/docs/sphinx/_build/html # Output directory
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating Sphinx HTML documentation"
            DEPENDS generate_doxygen_xml # Sphinx depends on Doxygen XML output
            VERBATIM
        )
  # Make sure sphinx_docs depends on doxygen_docs
  add_dependencies(build_sphinx_html generate_doxygen_xml)

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs/sphinx/_build/html/ DESTINATION share/doc/${PROJECT_NAME} COMPONENT Documentation)

endif() # End KAKUHEN_BUILD_DOCS
