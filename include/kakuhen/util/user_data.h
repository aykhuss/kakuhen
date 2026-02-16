#pragma once

#include "kakuhen/util/serialize.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file user_data.h
 * @brief Utilities for managing self-describing user-data records in binary files.
 *
 * Records are identified by a magic string and a keyword, allowing multiple
 * disparate data structures to be stored and retrieved from the same file.
 *
 * Layout:
 * [Magic] [Version:u8] [KeywordSize:u64] [PayloadSize:u64] [Keyword] [Payload]
 */

namespace kakuhen::util {

/// @brief Magic string identifying the start of a user-data record.
constexpr std::string_view USER_DATA_HEADER = "USERDATA";
/// @brief Current binary format version.
constexpr std::uint8_t USER_DATA_VERSION = 1;
/// @brief Defensive limit for keyword length to prevent excessive allocations.
constexpr std::size_t USER_DATA_MAX_KEYWORD_SIZE = 4096;

/// @brief Metadata for a discovered user-data record.
struct UserDataRecordInfo {
  std::string keyword;       ///< Unique identifier for the record.
  std::size_t payload_size;  ///< Size of the serialized payload in bytes.
  std::streamoff offset;     ///< Absolute file position of the record start.
};

namespace detail {

/// @brief Reads a 64-bit unsigned integer using the library's native binary format.
inline std::uint64_t read_u64(const char* ptr) {
  std::uint64_t v = 0;
  std::memcpy(&v, ptr, sizeof(v));
  return v;
}

/// @brief Safely converts u64 to size_t. Returns false on narrowing overflow.
inline bool to_size(std::uint64_t in, std::size_t& out) {
  out = static_cast<std::size_t>(in);
  return static_cast<std::uint64_t>(out) == in;
}

/// @brief Overflow-safe addition for size values.
inline bool checked_add(std::size_t a, std::size_t b, std::size_t& out) {
  if (b > std::numeric_limits<std::size_t>::max() - a) return false;
  out = a + b;
  return true;
}

/// @brief Fixed header size in bytes: magic + version + key_size + payload_size.
inline constexpr std::size_t fixed_header_size() {
  return USER_DATA_HEADER.size() + 1 + 8 + 8;
}

/**
 * @brief Attempts to read and validate a record header at the current stream position.
 * @return True if a valid header was parsed, filling @p info and @p record_size.
 *
 * On failure this function leaves the stream in a failed state; callers that continue
 * scanning should clear the stream state before the next operation.
 */
inline bool try_parse_header(std::istream& in, UserDataRecordInfo& info, std::size_t& record_size) {
  const std::streamoff start_pos = in.tellg();

  // 1. Check Magic
  std::array<char, USER_DATA_HEADER.size()> magic{};
  if (!in.read(magic.data(), static_cast<std::streamsize>(magic.size())) ||
      std::string_view(magic.data(), magic.size()) != USER_DATA_HEADER) {
    return false;
  }

  // 2. Read fixed-size metadata: Version(1), KeySize(8), PayloadSize(8)
  char meta[1 + 8 + 8];
  if (!in.read(meta, sizeof(meta))) return false;

  if (static_cast<std::uint8_t>(meta[0]) != USER_DATA_VERSION) return false;

  std::size_t kw_size = 0;
  std::size_t pl_size = 0;
  if (!to_size(read_u64(meta + 1), kw_size) || !to_size(read_u64(meta + 9), pl_size)) {
    return false;
  }

  if (kw_size > USER_DATA_MAX_KEYWORD_SIZE) return false;

  // 3. Read Keyword
  std::string keyword(kw_size, '\0');
  if (kw_size > 0 && !in.read(keyword.data(), static_cast<std::streamsize>(kw_size))) {
    return false;
  }

  info.keyword = std::move(keyword);
  info.payload_size = pl_size;
  info.offset = start_pos;
  std::size_t size = fixed_header_size();
  if (!checked_add(size, kw_size, size)) return false;
  if (!checked_add(size, pl_size, size)) return false;
  record_size = size;
  return true;
}

}  // namespace detail

/**
 * @brief Scans an input stream for all valid user-data records.
 * @param in The stream to scan.
 * @return A list of discovered record metadata in file order.
 * @throws std::runtime_error if stream size cannot be determined.
 */
[[nodiscard]] inline std::vector<UserDataRecordInfo> list_user_data_records_stream(
    std::istream& in) {
  const std::streampos origin = in.tellg();
  const auto restore = [&]() {
    in.clear();
    if (origin != std::streampos(-1)) in.seekg(origin);
  };
  in.clear();
  in.seekg(0, std::ios::end);
  const std::streampos end_pos = in.tellg();
  if (end_pos < 0) {
    restore();
    throw std::runtime_error("Failed to determine stream size");
  }
  const auto total_size = static_cast<std::size_t>(end_pos);

  std::vector<UserDataRecordInfo> records;
  if (total_size == 0) {
    restore();
    return records;
  }

  constexpr std::size_t CHUNK_SIZE = 65536;
  std::vector<char> buffer(CHUNK_SIZE);
  std::size_t stream_pos = 0;

  while (stream_pos < total_size) {
    in.clear();
    in.seekg(static_cast<std::streamoff>(stream_pos), std::ios::beg);
    in.read(buffer.data(), static_cast<std::streamsize>(CHUNK_SIZE));
    const std::size_t bytes_read = static_cast<std::size_t>(in.gcount());
    if (bytes_read == 0) break;

    const std::string_view view(buffer.data(), bytes_read);
    std::size_t search_pos = 0;

    bool advanced_to_next_record = false;
    while ((search_pos = view.find(USER_DATA_HEADER, search_pos)) != std::string_view::npos) {
      const std::size_t absolute_pos = stream_pos + search_pos;
      in.clear();
      in.seekg(static_cast<std::streamoff>(absolute_pos), std::ios::beg);

      UserDataRecordInfo info;
      std::size_t record_size = 0;
      if (detail::try_parse_header(in, info, record_size)) {
        std::size_t record_end = 0;
        if (!detail::checked_add(absolute_pos, record_size, record_end) ||
            record_end > total_size) {
          search_pos++;
          continue;
        }
        records.push_back(std::move(info));
        stream_pos = record_end;
        advanced_to_next_record = true;
        break;
      }
      search_pos++;
    }
    if (advanced_to_next_record) continue;

    // No headers found in this chunk, advance stream_pos.
    // We back up slightly to catch headers split across chunks.
    {
      const std::size_t overlap = USER_DATA_HEADER.size() - 1;
      stream_pos += (bytes_read > overlap) ? (bytes_read - overlap) : bytes_read;
    }
  }

  restore();
  return records;
}

/**
 * @brief Lists all user-data records in a file.
 * @throws std::ios_base::failure if the file cannot be opened.
 */
[[nodiscard]] inline std::vector<UserDataRecordInfo> list_user_data_records(
    const std::filesystem::path& filepath) {
  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::ios_base::failure("Failed to open file: " + filepath.string());
  }
  return list_user_data_records_stream(ifs);
}

/**
 * @brief Finds the start of a record with the specified keyword.
 * @param in The stream to search.
 * @param keyword The keyword to locate.
 * @return Absolute stream position, or -1 if not found.
 * @throws std::invalid_argument if keyword is empty.
 */
[[nodiscard]] inline std::streampos find_header_start(std::istream& in, std::string_view keyword) {
  if (keyword.empty()) throw std::invalid_argument("Keyword cannot be empty");
  for (const auto& record : list_user_data_records_stream(in)) {
    if (record.keyword == keyword) {
      in.clear();
      in.seekg(record.offset);
      return static_cast<std::streampos>(record.offset);
    }
  }
  return -1;
}

/**
 * @brief Serializes and writes a user-data record to a stream.
 * @throws std::invalid_argument if keyword is empty or too large.
 */
template <typename UD>
void write_user_data_stream(std::ostream& out, const UD& data,
                            std::string_view keyword = "USERDATA") {
  using namespace kakuhen::util::serialize;
  using DUD = std::decay_t<UD>;
  static_assert(std::is_trivially_copyable_v<DUD> || HasSerialize<DUD>,
                "UD must be trivially copyable or provide a serialize() method");
  if (keyword.empty() || keyword.size() > USER_DATA_MAX_KEYWORD_SIZE) {
    throw std::invalid_argument("Invalid user-data keyword");
  }

  std::ostringstream oss(std::ios::binary | std::ios::out);
  serialize_one<UD>(oss, data);
  const std::string payload = oss.str();

  write_bytes(out, USER_DATA_HEADER.data(), USER_DATA_HEADER.size());
  const std::uint8_t version = USER_DATA_VERSION;
  write_bytes(out, &version, sizeof(version));
  serialize_size(out, keyword.size());
  serialize_size(out, payload.size());
  write_bytes(out, keyword.data(), keyword.size());
  if (!payload.empty()) write_bytes(out, payload.data(), payload.size());
}

/**
 * @brief Reads a user-data record from a stream.
 * @throws std::invalid_argument if keyword is empty or too large.
 * @throws std::runtime_error if the header is malformed or keyword mismatches.
 */
template <typename UD>
void read_user_data_stream(std::istream& in, UD& data, std::string_view keyword = "USERDATA") {
  using namespace kakuhen::util::serialize;
  using DUD = std::decay_t<UD>;
  static_assert(std::is_trivially_copyable_v<DUD> || HasDeserialize<DUD>,
                "UD must be trivially copyable or provide a deserialize() method");
  if (keyword.empty() || keyword.size() > USER_DATA_MAX_KEYWORD_SIZE) {
    throw std::invalid_argument("Invalid user-data keyword");
  }

  UserDataRecordInfo info;
  std::size_t record_size = 0;

  if (!detail::try_parse_header(in, info, record_size)) {
    throw std::runtime_error("Failed to parse user-data record header");
  }

  if (info.keyword != keyword) {
    throw std::runtime_error("Incompatible user data headers");
  }

  std::string payload(info.payload_size, '\0');
  if (info.payload_size > 0 &&
      !in.read(payload.data(), static_cast<std::streamsize>(info.payload_size))) {
    throw std::runtime_error("Failed to read user-data payload");
  }

  std::istringstream iss(payload, std::ios::binary | std::ios::in);
  deserialize_one<UD>(iss, data);
  if (iss.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("User-data payload not fully consumed during deserialization");
  }
}

/**
 * @brief Appends a record to a file, ensuring the keyword is unique.
 * @param filepath Path to the target file.
 * @param data Payload to append.
 * @param keyword Record keyword.
 * @throws std::runtime_error if the keyword already exists in the file.
 * @throws std::ios_base::failure if file operations fail.
 */
template <typename UD>
void write_user_data(const std::filesystem::path& filepath, const UD& data,
                     std::string_view keyword = "USERDATA") {
  if (std::filesystem::exists(filepath)) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (ifs && find_header_start(ifs, keyword) != -1) {
      throw std::runtime_error("Keyword '" + std::string(keyword) + "' already exists in " +
                               filepath.string());
    }
  }

  std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
  if (!ofs) throw std::ios_base::failure("Failed to open file for writing: " + filepath.string());
  write_user_data_stream(ofs, data, keyword);
}

/**
 * @brief Reads a record from a file by keyword.
 * @param filepath Path to the source file.
 * @param data Destination object.
 * @param keyword Record keyword.
 * @throws std::runtime_error if the keyword is not found.
 * @throws std::ios_base::failure if file operations fail.
 */
template <typename UD>
void read_user_data(const std::filesystem::path& filepath, UD& data,
                    std::string_view keyword = "USERDATA") {
  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs) throw std::ios_base::failure("Failed to open file for reading: " + filepath.string());
  if (find_header_start(ifs, keyword) == -1) {
    throw std::runtime_error("Keyword '" + std::string(keyword) + "' not found in " +
                             filepath.string());
  }
  read_user_data_stream(ifs, data, keyword);
}

}  // namespace kakuhen::util
