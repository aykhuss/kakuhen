#pragma once

// shared helpers for tests that touch the filesystem

#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace test_support {

/// fresh scratch directory, removed with everything in it on destruction
struct TempDir {
  std::filesystem::path dir;

  explicit TempDir(std::string_view prefix = "kakuhen-test")
      : dir(std::filesystem::temp_directory_path() /
            (std::string(prefix) + "-" + std::to_string(std::random_device{}()))) {
    if (!std::filesystem::create_directory(dir)) throw std::runtime_error("test directory exists");
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
};

inline std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

inline void write_file(const std::filesystem::path& path, std::string_view bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace test_support
