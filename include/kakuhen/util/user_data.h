#pragma once

#include "kakuhen/util/serialize.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace kakuhen::util {

constexpr std::string_view USER_DATA_HEADER = "USER_DATA_HEADER";

/// write custom user data to an output stream
template <typename UD>
void write_user_data_stream(std::ostream& out, const UD& user_data,
                            const std::string_view header = "USERDATA") {
  using namespace kakuhen::util::serialize;
  if (header.empty()) throw std::invalid_argument("Header cannot be empty");
  const std::string pattern(std::string(USER_DATA_HEADER) + std::string(header));
  write_bytes(out, pattern.data(), pattern.size());
  serialize_one<UD>(out, user_data);
}

/// read custom user data from an input stream
template <typename UD>
void read_user_data_stream(std::istream& in, UD& user_data,
                           const std::string_view header = "USERDATA") {
  using namespace kakuhen::util::serialize;
  if (header.empty()) throw std::invalid_argument("Header cannot be empty");
  const std::string pattern(std::string(USER_DATA_HEADER) + std::string(header));
  /// check that the header matches
  std::vector<char> buf(pattern.size());
  read_bytes(in, buf.data(), pattern.size());
  if (std::string_view(buf.data(), buf.size()) != pattern) {
    throw std::runtime_error("Incompatible user data headers");
  }
  /// read in the user data
  deserialize_one<UD>(in, user_data);
}

/// helper function to find the start of a pattern in an input stream
inline std::streampos find_pattern_start(std::istream& in, std::string_view pattern) {
  if (!in) throw std::runtime_error("Invalid input stream");
  if (pattern.empty()) throw std::runtime_error("Empty pattern");

  const size_t pat_len = pattern.size();
  std::vector<char> buffer(4096);
  std::string window;
  window.reserve(buffer.size() + pat_len);

  std::streampos base_pos = in.tellg();

  while (in.read(buffer.data(), buffer.size()) || in.gcount() > 0) {
    size_t bytes_read = in.gcount();
    window.append(buffer.data(), bytes_read);

    auto it = std::search(window.begin(), window.end(), pattern.begin(), pattern.end());

    if (it != window.end()) {
      std::streampos found_pos = base_pos + std::streamoff(std::distance(window.begin(), it));
      in.clear();
      in.seekg(found_pos);
      return found_pos;
    }

    if (window.size() > pat_len) {
      base_pos += std::streamoff(window.size() - pat_len);
      window.erase(0, window.size() - pat_len);
    }
  }

  in.clear();
  in.seekg(0, std::ios::end);
  return -1;
}

/// wrapper that includes the header prefix in the search
inline std::streampos find_header_start(std::istream& in, std::string_view header) {
  const std::string pattern(std::string(USER_DATA_HEADER) + std::string(header));
  return find_pattern_start(in, pattern);
}

/// write custom user data to a file
template <typename UD>
void write_user_data(const std::filesystem::path& filepath, const UD& user_data,
                     const std::string_view header = "USERDATA") {
  /// chec for existing header
  if (std::filesystem::exists(filepath)) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::ios_base::failure("Failed to open file: " + filepath.string());
    }
    std::streampos pos = find_header_start(ifs, header);
    if (pos != -1) {
      throw std::runtime_error("Header already exists in file " + filepath.string());
    }
  }  // ifs closes here
  /// append user data to the end of the file
  std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
  if (!ofs.is_open()) {
    throw std::ios_base::failure("Failed to open file: " + filepath.string());
  }
  write_user_data_stream(ofs, user_data, header);
  if (!ofs) {
    throw std::ios_base::failure("Error writing user data to file: " + filepath.string());
  }
}

/// read custom user data from a file
template <typename UD>
void read_user_data(const std::filesystem::path& filepath, UD& user_data,
                    const std::string_view header = "USERDATA") {
  /// first check if the header actually appears in the file
  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::ios_base::failure("Failed to open file: " + filepath.string());
  }
  std::streampos pos = find_header_start(ifs, header);
  if (pos == -1) {
    throw std::runtime_error("Header not found in file " + filepath.string());
  }
  ifs.seekg(pos);
  read_user_data_stream(ifs, user_data, header);
}

}  // namespace kakuhen::util
