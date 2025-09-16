include(FetchContent)

#> argparse for CLI argument parsing
FetchContent_Declare(
  argparse
  # GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  URL https://github.com/p-ranav/argparse/archive/refs/tags/v3.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(argparse)

#> library to handle JSON data structure
FetchContent_Declare(
  json
  # GIT_REPOSITORY https://github.com/nlohmann/json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

add_executable(kakuhen_cli kakuhen_cli.cpp)

target_link_libraries(kakuhen_cli
  PRIVATE
    kakuhen
    argparse
    nlohmann_json::nlohmann_json
)
set_target_properties(kakuhen_cli PROPERTIES OUTPUT_NAME "kakuhen")
install(TARGETS kakuhen_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
