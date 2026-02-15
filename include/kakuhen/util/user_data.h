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

/// @brief Fixed header prefix prepended to all user data serialization.
constexpr std::string_view USER_DATA_HEADER = "USER_DATA_HEADER";

/*!
 * @brief Writes custom user data to an output stream with a specific header.
 *
 * This function serializes a user-defined data structure (`UD`) to an output
 * stream, prefixed with a unique header to enable identification and integrity
 * checking upon deserialization.
 *
 * @tparam UD The type of the user-defined data.
 * @param out The output stream to write to.
 * @param user_data The user-defined data object to serialize.
 * @param header A string view representing the custom header. Defaults to "USERDATA".
 * @throws std::invalid_argument if the header is empty.
 */
template <typename UD>
void write_user_data_stream(std::ostream& out, const UD& user_data,
                            const std::string_view header = "USERDATA") {
  using namespace kakuhen::util::serialize;
  using DUD = std::decay_t<UD>;
  static_assert(std::is_trivially_copyable_v<DUD> || HasSerialize<DUD>,
                "UD must be trivially copyable or provide a serialize() method");

  if (header.empty()) throw std::invalid_argument("Header cannot be empty");

  std::string pattern;
  pattern.reserve(USER_DATA_HEADER.size() + header.size());
  pattern.append(USER_DATA_HEADER);
  pattern.append(header);

  write_bytes(out, pattern.data(), pattern.size());
  serialize_one<UD>(out, user_data);
}

/*!
 * @brief Reads custom user data from an input stream with a specific header.
 *
 * This function deserializes a user-defined data structure (`UD`) from an
 * input stream, verifying that the expected header is present to ensure
 * data consistency.
 *
 * @tparam UD The type of the user-defined data.
 * @param in The input stream to read from.
 * @param user_data The user-defined data object to deserialize into.
 * @param header A string view representing the custom header. Defaults to "USERDATA".
 * @throws std::invalid_argument if the header is empty.
 * @throws std::runtime_error if the header in the stream does not match the expected header.
 */
template <typename UD>
void read_user_data_stream(std::istream& in, UD& user_data,
                           const std::string_view header = "USERDATA") {
  using namespace kakuhen::util::serialize;
  using DUD = std::decay_t<UD>;
  static_assert(std::is_trivially_copyable_v<DUD> || HasDeserialize<DUD>,
                "UD must be trivially copyable or provide a deserialize() method");

  if (header.empty()) throw std::invalid_argument("Header cannot be empty");

  std::string pattern;
  pattern.reserve(USER_DATA_HEADER.size() + header.size());
  pattern.append(USER_DATA_HEADER);
  pattern.append(header);

  // check that the header matches
  std::vector<char> buf(pattern.size());
  read_bytes(in, buf.data(), pattern.size());
  if (std::string_view(buf.data(), buf.size()) != pattern) {
    throw std::runtime_error("Incompatible user data headers");
  }
  // read in the user data
  deserialize_one<UD>(in, user_data);
}

/*!
 * @brief Finds the starting position of a given pattern in an input stream.
 *
 * This function searches through the input stream for the first occurrence
 * of `pattern`. If found, the stream's read pointer is set to the beginning
 * of the pattern.
 *
 * @tparam BufferSize The size of the read buffer. Defaults to 4096 bytes.
 * @param in The input stream to search within.
 * @param pattern The byte sequence pattern to find.
 * @return The stream position where the pattern starts, or -1 if not found.
 * @throws std::runtime_error if the input stream is invalid.
 * @throws std::runtime_error if the pattern is empty.
 */
template <size_t BufferSize = 4096>
inline std::streampos find_pattern_start(std::istream& in, std::string_view pattern) {
  if (!in) throw std::runtime_error("Invalid input stream");
  if (pattern.empty()) throw std::runtime_error("Empty pattern");

  const size_t pat_len = pattern.size();
  std::vector<char> buffer(BufferSize);
  std::string window;
  window.reserve(buffer.size() + pat_len);

  std::streampos base_pos = in.tellg();

  while (in.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) || in.gcount() > 0) {
    size_t bytes_read = static_cast<size_t>(in.gcount());
    window.append(buffer.data(), bytes_read);

    auto it = std::search(window.begin(), window.end(), pattern.begin(), pattern.end());

    if (it != window.end()) {
      std::streampos found_pos = base_pos + std::streamoff(std::distance(window.begin(), it));
      in.clear();
      in.seekg(found_pos);
      return found_pos;
    }

    // Keep the overlap (size of pattern) for the next iteration to detect split patterns
    if (window.size() > pat_len) {
      base_pos += std::streamoff(window.size() - pat_len);
      window.erase(0, window.size() - pat_len);
    }
  }

  in.clear();
  in.seekg(0, std::ios::end);
  return -1;
}

/*!
 * @brief Helper function to find the start of a user data header in a stream.
 *
 * This function constructs the full pattern (USER_DATA_HEADER + custom header)
 * and uses `find_pattern_start` to locate it in the stream.
 *
 * @param in The input stream to search within.
 * @param header The custom header string view.
 * @return The stream position where the header starts, or -1 if not found.
 */
inline std::streampos find_header_start(std::istream& in, std::string_view header) {
  std::string pattern;
  pattern.reserve(USER_DATA_HEADER.size() + header.size());
  pattern.append(USER_DATA_HEADER);
  pattern.append(header);
  return find_pattern_start(in, pattern);
}

/*!
 * @brief Writes custom user data to a file, appending it if the file exists.
 *
 * This function handles writing user-defined data to a specified file. It checks
 * if the header already exists in the file and throws an error if it does,
 * preventing duplicate data. Otherwise, it appends the data to the end of the file.
 *
 * @tparam UD The type of the user-defined data.
 * @param filepath The path to the file.
 * @param user_data The user-defined data object to serialize.
 * @param header A string view representing the custom header. Defaults to "USERDATA".
 * @throws std::ios_base::failure if the file cannot be opened.
 * @throws std::runtime_error if the header already exists in the file.
 * @throws std::ios_base::failure if there's an error writing to the file.
 */
template <typename UD>
void write_user_data(const std::filesystem::path& filepath, const UD& user_data,
                     const std::string_view header = "USERDATA") {
  // check for existing header
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
  // append user data to the end of the file
  std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
  if (!ofs.is_open()) {
    throw std::ios_base::failure("Failed to open file: " + filepath.string());
  }
  write_user_data_stream(ofs, user_data, header);
  if (!ofs) {
    throw std::ios_base::failure("Error writing user data to file: " + filepath.string());
  }
}

/*!
 * @brief Reads custom user data from a file with a specific header.
 *
 * This function locates the specified header within a file and then
 * deserializes the associated user-defined data.
 *
 * @tparam UD The type of the user-defined data.
 * @param filepath The path to the file.
 * @param user_data The user-defined data object to deserialize into.
 * @param header A string view representing the custom header. Defaults to "USERDATA".
 * @throws std::ios_base::failure if the file cannot be opened.
 * @throws std::runtime_error if the header is not found in the file.
 */
template <typename UD>
void read_user_data(const std::filesystem::path& filepath, UD& user_data,
                    const std::string_view header = "USERDATA") {
  // first check if the header actually appears in the file
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
